#!/usr/bin/env bash
set -e
BASE=/work3/s241614/runs/train/y11s_prod_tune2/weights/best.pt
yolo train \
  model=$BASE \
  data=/work3/s241614/projects/yolo_dataset/data.yaml \
  imgsz=960 \
  epochs=30 \
  batch=32 device=0 workers=8 \
  lr0=0.0005 lrf=0.1 optimizer=AdamW weight_decay=0.01 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 erasing=0.1 \
  degrees=0.0 shear=0.0 translate=0.02 scale=0.2 fliplr=0.2 \
  rect=False cache=disk deterministic=True seed=0 \
  patience=50 \
  project=/work3/s241614/runs/train name=y11s_postfinetune
