# Reproducerbarhed
## Basismodel
yolo train model=yolo11s.pt data=/work3/s241614/projects/yolo_dataset/data.yaml imgsz=832 epochs=150 batch=32 \
  optimizer=AdamW lr0=0.002 lrf=0.15 weight_decay=0.01 cos_lr=True warmup_epochs=5 \
  mosaic=0.8 close_mosaic=25 mixup=0.05 copy_paste=0.3 erasing=0.2 \
  degrees=3 shear=1.0 translate=0.05 scale=0.5 fliplr=0.3 \
  cls=1.2 box=7.5 dfl=1.5 kobj=1.0 \
  project=/work3/s241614/runs/train name=y11s_prod_tune2

## Finetune
yolo train model=/work3/s241614/runs/train/y11s_prod_tune2/weights/best.pt data=/work3/s241614/projects/yolo_dataset/data.yaml imgsz=960 epochs=30 batch=32 \
  optimizer=AdamW lr0=0.0008 lrf=0.1 weight_decay=0.01 cos_lr=True warmup_epochs=3 \
  mosaic=0.8 close_mosaic=20 mixup=0.05 copy_paste=0.3 erasing=0.2 \
  degrees=3 shear=1.0 translate=0.05 scale=0.5 fliplr=0.3 \
  cls=1.2 box=7.5 dfl=1.5 kobj=1.0 \
  rect=False cache=disk deterministic=True seed=0 \
  project=/work3/s241614/runs/train name=y11s_postfinetune

## Validering
yolo val model=/work3/s241614/runs/train/y11s_postfinetune/weights/best.pt data=/work3/s241614/projects/yolo_dataset/data.yaml imgsz=960 batch=32

## Eksport
yolo export model=/work3/s241614/runs/train/y11s_postfinetune/weights/best.pt format=onnx
yolo export model=/work3/s241614/runs/train/y11s_postfinetune/weights/best.pt format=torchscript
