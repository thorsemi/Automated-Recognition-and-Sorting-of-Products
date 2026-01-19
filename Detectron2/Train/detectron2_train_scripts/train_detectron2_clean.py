#!/usr/bin/env python3
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_  # numpy>=1.24 kompatibilitet

import os, random, numpy as np, torch
from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

# ----------------- DATASETS -----------------
DATASET_ROOT = Path("/work3/s241614/datasets/custom_dataset")
TRAIN_JSON = DATASET_ROOT/"annotations/instances_train.json"
VAL_JSON   = DATASET_ROOT/"annotations/instances_val.json"
TRAIN_IMG  = DATASET_ROOT/"train"
VAL_IMG    = DATASET_ROOT/"val"

# Registrer (idempotent)
try:
    register_coco_instances("leaflet_train", {}, str(TRAIN_JSON), str(TRAIN_IMG))
    register_coco_instances("leaflet_val",   {}, str(VAL_JSON),   str(VAL_IMG))
except KeyError:
    pass

# Debug – se at datasets er til stede
print("DEBUG datasets:", DatasetCatalog.list())
print("DEBUG leaflet_train size:", len(DatasetCatalog.get("leaflet_train")))
print("DEBUG leaflet_val   size:", len(DatasetCatalog.get("leaflet_val")))

# ----------------- SEED -----------------
seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

# ----------------- CFG -----------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
)

cfg.DATASETS.TRAIN = ("leaflet_train",)
cfg.DATASETS.TEST  = ("leaflet_val",)

cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # dine 7 klasser
cfg.MODEL.DEVICE = "cuda"

# Små, sikre værdier til at komme i gang
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 2000
cfg.SOLVER.STEPS = []   # ingen LR-step (vores MAX_ITER er lille)
cfg.SOLVER.WARMUP_ITERS = 0

cfg.TEST.EVAL_PERIOD = 0
cfg.OUTPUT_DIR = "./out-clean"

# ----------------- TRAIN -----------------
Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()