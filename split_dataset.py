import os
import shutil
import random
from pathlib import Path

# Paths
SOURCE_DIR = 'dataset'
DEST_DIR = 'dataset_split'
CLASSES = ['1_normal', '2_cataract', '2_glaucoma', '3_retina_disease']

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create split folders
for split in ['train', 'val', 'test']:
    for cls in CLASSES:
        Path(f"{DEST_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Go through each class
for cls in CLASSES:
    class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Helper to copy images
    def copy_images(img_list, split):
        for img in img_list:
            src = os.path.join(class_path, img)
            dst = os.path.join(DEST_DIR, split, cls, img)
            shutil.copy(src, dst)

    # Copy images
    copy_images(train_images, 'train')
    copy_images(val_images, 'val')
    copy_images(test_images, 'test')

print("✅ Dataset split completed!")
