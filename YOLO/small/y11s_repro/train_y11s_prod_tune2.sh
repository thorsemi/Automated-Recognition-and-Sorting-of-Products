#!/usr/bin/env bash
set -e
MODEL=yolo11s.pt
yolo train \
  model=$MODEL \
  data=/work3/s241614/projects/yolo_dataset/data.yaml \
  imgsz=832 \
  epochs=150 \
  batch=32 device=0 workers=8 \
  cos_lr=True warmup_epochs=5 \
  optimizer=AdamW lr0=0.002 lrf=0.15 weight_decay=0.01 \
  hsv_h=0.008 hsv_s=0.5 hsv_v=0.3 \
  mosaic=0.8 close_mosaic=25 mixup=0.05 copy_paste=0.3 erasing=0.2 \
  degrees=3 shear=1.0 translate=0.05 scale=0.5 fliplr=0.3 \
  rect=False cache=disk deterministic=True seed=0 \
  cls=1.2 box=7.5 dfl=1.5 kobj=1.0 \
  project=/work3/s241614/runs/train name=y11s_prod_tune2
