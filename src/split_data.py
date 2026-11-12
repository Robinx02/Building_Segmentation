import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ----------------------------------------------------------------------------- io
def list_pairs(root: Path):
    
    """Return sorted list of image file names that have a matching label (same stem)."""
    
    img_dir, lbl_dir = root / "image", root / "label"
    lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    names, missing = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        (names if p.stem in lbl_stems else missing).append(p.name)
    if missing:
        print(f"[warn] {len(missing)} images have no label and are skipped, e.g. {missing[:3]}")
    return names


def read_rgb_small(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


# ------------------------------------------------------------------ union-find
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# --------------------------------------------------------------- grouping rules
def groups_from_filename(names, regex):
    pat = re.compile(regex)
    dsu = DSU(len(names))
    keys = {}
    for i, n in enumerate(names):
        m = pat.search(Path(n).stem)
        key = m.group(1) if m and m.groups() else Path(n).stem
        if key in keys:
            dsu.union(keys[key], i)
        else:
            keys[key] = i
    return dsu


def link_adjacent(names, img_dir: Path, dsu: DSU, ratio_thr=1.5, verbose=True):
    """
    Two tiles are neighbours if the last column/row of A continues into the first
    column/row of B. Seam difference is compared with the typical difference between
    neighbouring columns *inside* the tiles (ratio ~1 for true neighbours, >>1 otherwise).
    Only the best match per tile and direction is considered (mutual-best check).
    NOTE: finds touching tiles. Tiles that overlap each other are NOT detected.
    """
    imgs = np.stack([read_rgb_small(img_dir / n) for n in names]).astype(np.float32)  # N,H,W,3
    N = len(names)

    def seam_links(first, last, inner_a, inner_b):
        # first/last: (N, L*3) borders of each tile; inner_*: reference intra-tile diffs per tile
        # distance matrix D[i,j] = mean |last_i - first_j|
        D = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            D[i] = np.abs(first - last[i]).mean(axis=1)
        np.fill_diagonal(D, np.inf)
        ref = 0.5 * (inner_a[:, None] + inner_b[None, :]) + 1e-3
        R = D / ref
        links = []
        best_j = R.argmin(axis=1)
        best_i = R.argmin(axis=0)
        for i in range(N):
            j = best_j[i]
            if R[i, j] < ratio_thr and best_i[j] == i:
                links.append((i, j, float(R[i, j])))
        return links

    # horizontal: right border of i -> left border of j
    right = imgs[:, :, -1, :].reshape(N, -1)
    left = imgs[:, :, 0, :].reshape(N, -1)
    in_r = np.abs(imgs[:, :, -1, :] - imgs[:, :, -2, :]).mean(axis=(1, 2))
    in_l = np.abs(imgs[:, :, 0, :] - imgs[:, :, 1, :]).mean(axis=(1, 2))
    # vertical: bottom border of i -> top border of j
    bottom = imgs[:, -1, :, :].reshape(N, -1)
    top = imgs[:, 0, :, :].reshape(N, -1)
    in_b = np.abs(imgs[:, -1, :, :] - imgs[:, -2, :, :]).mean(axis=(1, 2))
    in_t = np.abs(imgs[:, 0, :, :] - imgs[:, 1, :, :]).mean(axis=(1, 2))

    links = seam_links(left, right, in_r, in_l) + seam_links(top, bottom, in_b, in_t)
    for i, j, _ in links:
        dsu.union(i, j)
    if verbose:
        print(f"[adjacency] found {len(links)} touching tile pairs")
        if not links:
            print("            -> tiles look independent (e.g. different cities). ""Use --group_by cluster to also separate similar-looking scenes.")
    return dsu


def link_by_appearance(names, img_dir: Path, dsu: DSU, n_clusters=None, seed=42):
    """Cluster tiles by colour statistics (proxy for city / sensor / season)."""
    from sklearn.cluster import KMeans

    feats = []
    for n in names:
        img = read_rgb_small(img_dir / n)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        f = []
        for c in range(3):
            h = np.histogram(img[..., c], bins=16, range=(0, 256))[0].astype(np.float32)
            f.append(h / h.sum())
        f.append(img.reshape(-1, 3).mean(0) / 255.0)
        f.append(img.reshape(-1, 3).std(0) / 255.0)
        f.append(hsv.reshape(-1, 3).mean(0) / np.array([180, 255, 255]))
        feats.append(np.concatenate(f))
    X = np.stack(feats)
    k = n_clusters or max(6, len(names) // 12)  # ~12 tiles per cluster by default
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    first = {}
    for i, l in enumerate(labels):
        if l in first:
            dsu.union(first[l], i)
        else:
            first[l] = i
    print(f"[cluster] {k} appearance clusters")
    return dsu


# splitting
def split_groups(groups, n_val, n_test, seed):
    """
    groups: dict gid -> [file names]. Whole groups stay together. Largest groups are
    placed first, each into the split that is currently furthest below its target size
    (train target = N - n_val - n_test).
    """
    rng = random.Random(seed)
    gids = list(groups)
    rng.shuffle(gids)  # random tie-break between equally sized groups
    gids.sort(key=lambda g: -len(groups[g]))
    n_total = sum(len(v) for v in groups.values())
    target = {"train": n_total - n_val - n_test, "val": n_val, "test": n_test}
    split = {"train": [], "val": [], "test": []}
    for g in gids:
        s = max(split, key=lambda k: target[k] - len(split[k]))
        split[s] += groups[g]
    for s, files in split.items():
        if abs(len(files) - target[s]) > max(3, 0.25 * target[s]):
            print(f"[warn] {s} has {len(files)} images (target {target[s]}): groups are large. ""Use a weaker grouping (e.g. adjacency only) or accept the imbalance.")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw", help="folder containing image/ and label/")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--n_val", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_by", default="adjacency",
                    choices=["filename", "adjacency", "cluster", "adjacency+cluster", "none"])
    ap.add_argument("--group_regex", default=r"^(.*?)(?:[_-]?\d+)?$",
                    help="regex with ONE capture group giving the group key from the file stem")
    ap.add_argument("--adj_thresh", type=float, default=1.5, help="seam/inner-diff ratio for adjacency")
    ap.add_argument("--n_clusters", type=int, default=None)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list_pairs(root)
    print(f"{len(names)} image/label pairs found")
    assert len(names) > 0, "no pairs found - check --root (expects <root>/image and <root>/label)"

    dsu = DSU(len(names))
    if args.group_by == "filename":
        dsu = groups_from_filename(names, args.group_regex)
    if args.group_by in ("adjacency", "adjacency+cluster"):
        dsu = link_adjacent(names, root / "image", dsu, args.adj_thresh)
    if args.group_by in ("cluster", "adjacency+cluster"):
        dsu = link_by_appearance(names, root / "image", dsu, args.n_clusters, args.seed)
    if args.group_by == "none":
        print("[warn] group_by=none -> plain random split, spatial leakage NOT controlled")

    groups = defaultdict(list)
    for i, n in enumerate(names):
        groups[dsu.find(i)].append(n)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"{len(groups)} groups, largest sizes: {sizes[:10]}")

    split = split_groups(groups, args.n_val, args.n_test, args.seed)

    # ---- sanity checks: disjoint splits, nothing lost, no group straddles two splits
    allf = split["train"] + split["val"] + split["test"]
    assert len(allf) == len(set(allf)) == len(names), "files lost or duplicated"
    gid_of = {n: dsu.find(i) for i, n in enumerate(names)}
    for g, files in groups.items():
        owners = {s for s in split for f in files if f in split[s]}
        assert len(owners) == 1, f"group {g} leaks across splits {owners}"

    for s, files in split.items():
        (out / f"{s}.txt").write_text("\n".join(sorted(files)) + "\n")
    with open(out / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "group"])
        for s, files in split.items():
            for n in sorted(files):
                w.writerow([n, s, gid_of[n]])

    print({s: len(v) for s, v in split.items()})
    print(f"wrote {out}/train.txt val.txt test.txt split_manifest.csv")


if __name__ == "__main__":
    main()