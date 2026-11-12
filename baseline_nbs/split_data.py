import os

# Paths
image_dir = "data/raw/satellite_dataset_I/image"
splits_dir = "data/splits"

# Get sorted images to prevent spatial leakage (looking for .tif)
images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff'))])

# Ensure we found the images
if not images:
    print("Error: No .tif files found in", image_dir)
    exit()

# Split into 160 train, 24 val, 20 test
train_imgs = images[:160]
val_imgs = images[160:184]
test_imgs = images[184:]

# Save to text files
with open(os.path.join(splits_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_imgs))
with open(os.path.join(splits_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_imgs))
with open(os.path.join(splits_dir, "test.txt"), "w") as f:
    f.write("\n".join(test_imgs))

print(f"Found {len(images)} images. Splits successfully created in data/splits/")