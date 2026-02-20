import os
import shutil
import random

BASE_DIR = "../data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

NEW_VAL_RATIO = 0.15  # 15%

classes = ["NORMAL", "PNEUMONIA"]

os.makedirs(VAL_DIR, exist_ok=True)

for cls in classes:
    src_dir = os.path.join(TRAIN_DIR, cls)
    val_cls_dir = os.path.join(VAL_DIR, cls)

    os.makedirs(val_cls_dir, exist_ok=True)

    files = os.listdir(src_dir)
    random.shuffle(files)

    val_size = int(len(files) * NEW_VAL_RATIO)
    val_files = files[:val_size]

    for f in val_files:
        shutil.move(
            os.path.join(src_dir, f),
            os.path.join(val_cls_dir, f)
        )

    print(f"{cls}: moved {len(val_files)} files to val/")
