"""Train U-Net on WHU, pick best epoch on VAL, evaluate once on TEST, write results to CSV."""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import WHUDataset, compute_mean_std, read_manifest
from loss import get_loss
from metrics import evaluate_masks, summarize, result_row, append_csv, write_rows_csv
from unet_arch import UNet


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, criterion, device, amp, thr=0.5):
    """Returns mean loss and a list of (name, binary_pred, gt) per image."""
    model.eval()
    tot, n, out = 0.0, 0, []
    for x, y, names in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        tot += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
        gts = y.cpu().numpy()[:, 0]
        for nm, p, g in zip(names, probs, gts):
            out.append((nm, (p > thr).astype(np.uint8), g.astype(np.uint8)))
    return tot / max(n, 1), out


def score(preds, tol):
    rows = evaluate_masks([p[0] for p in preds], [p[1] for p in preds], [p[2] for p in preds], tol)
    return rows, summarize(rows)


def save_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    for k, lab in [("val_iou", "IoU"), ("val_dice", "Dice"), ("val_boundary_f1", "Boundary F1")]:
        ax[1].plot(ep, [h[k] for h in hist], label=lab)
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Validation metrics")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_masks(preds, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for nm, p, _ in preds:
        cv2.imwrite(str(folder / f"{Path(nm).stem}.png"), p * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--loss", default="bce_dice", choices=["bce_dice", "bce_iou", "boundary"])
    ap.add_argument("--edge_weight", type=float, default=5.0)
    ap.add_argument("--edge_width", type=int, default=2)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256, help="random train crop size (0 = full tile)")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--bf_tol", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--bilinear", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = args.run_name or f"unet_{args.loss}"
    out = Path(args.out_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"run={run} device={device} amp={amp}")

    # data (normalisation stats from train only)
    train_names = read_manifest(Path(args.splits) / "train.txt")
    val_names = read_manifest(Path(args.splits) / "val.txt")
    test_names = read_manifest(Path(args.splits) / "test.txt")
    assert not (set(train_names) & set(val_names) | set(train_names) & set(test_names)
                | set(val_names) & set(test_names)), "splits overlap!"
    print(f"train/val/test = {len(train_names)}/{len(val_names)}/{len(test_names)}")

    mean, std = compute_mean_std(args.data_root, train_names)
    print("norm mean/std:", np.round(mean, 4), np.round(std, 4))

    ds_tr = WHUDataset(args.data_root, train_names, train=True, crop=args.crop, mean=mean, std=std,
                    repeat=args.repeat)
    ds_va = WHUDataset(args.data_root, val_names, mean=mean, std=std)
    ds_te = WHUDataset(args.data_root, test_names, mean=mean, std=std)
    g = torch.Generator().manual_seed(args.seed)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True, generator=g, **kw)
    dl_va = DataLoader(ds_va, batch_size=4, shuffle=False, **kw)
    dl_te = DataLoader(ds_te, batch_size=4, shuffle=False, **kw)

    # model / loss / optimiser
    model = UNet(3, 1, bilinear=args.bilinear).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"unet: {n_params / 1e6:.1f}M parameters")
    criterion = get_loss(args.loss, args.edge_weight, args.edge_width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt = out / "checkpoints" / f"{run}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    hist, best, best_ep, bad = [], -1.0, 0, 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            nb += 1

        vloss, vpreds = predict(model, dl_va, criterion, device, amp, args.thr)
        _, vs = score(vpreds, args.bf_tol)
        sched.step(vs["iou"])
        h = dict(epoch=ep, train_loss=tl / max(nb, 1), val_loss=vloss, val_iou=vs["iou"],
                val_dice=vs["dice"], val_precision=vs["precision"], val_recall=vs["recall"],
                val_boundary_f1=vs["boundary_f1"], lr=opt.param_groups[0]["lr"],
                elapsed_s=round(time.time() - t0))
        hist.append(h)

        flag = ""
        if vs["iou"] > best:
            best, best_ep, bad = vs["iou"], ep, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std,
                        "epoch": ep}, ckpt)
            flag = "  *best*"
        else:
            bad += 1
        print(f"ep {ep:3d} | train {h['train_loss']:.4f} | val {vloss:.4f} | IoU {vs['iou']:.4f} "
            f"Dice {vs['dice']:.4f} BF1 {vs['boundary_f1']:.4f}{flag}")
        if bad >= args.patience:
            print(f"early stop (no val IoU gain for {args.patience} epochs)")
            break

    write_rows_csv(out / "history" / f"{run}.csv", hist)
    save_curves(hist, out / "plots" / f"{run}_curves.png")

    # final evaluation with best-on-val weights
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    extra = dict(loss=args.loss, edge_weight=args.edge_weight if args.loss == "boundary" else "",
                edge_width=args.edge_width if args.loss == "boundary" else "",
                bf_tol_px=args.bf_tol, best_epoch=best_ep, epochs_run=len(hist), seed=args.seed,
                params_M=round(n_params / 1e6, 2), crop=args.crop)
    for split, dl in [("val", dl_va), ("test", dl_te)]:
        _, preds = predict(model, dl, criterion, device, amp, args.thr)
        rows, summ = score(preds, args.bf_tol)
        write_rows_csv(out / "per_image" / f"{run}_{split}.csv", rows)
        append_csv(out / "metrics.csv", result_row(run, split, summ, **extra))
        save_masks(preds, out / "predictions" / run / split)
        print(f"[{split}] " + " ".join(f"{k}={summ[k]:.4f}" for k in
                                    ["iou", "dice", "precision", "recall", "boundary_f1"]))
    print(f"done. results -> {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()