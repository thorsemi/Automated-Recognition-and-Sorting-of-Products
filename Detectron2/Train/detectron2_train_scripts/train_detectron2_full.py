
#!/usr/bin/env python3
import os, random, json
from pathlib import Path
import numpy as np
import torch

# --- NumPy 2.0 shim (undgår np.bool-fejl i detectron2/pycocotools) ---
try:
    _ = np.bool
except AttributeError:
    np.bool = bool  # type: ignore[attr-defined]

from detectron2.engine import DefaultTrainer, hooks
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# ----------------- DATASETS -----------------
DATASET_ROOT = Path("/work3/s241614/datasets/custom_dataset")
TRAIN_JSON = DATASET_ROOT / "annotations/instances_train.json"
VAL_JSON   = DATASET_ROOT / "annotations/instances_val.json"
TRAIN_IMG  = DATASET_ROOT / "train"
VAL_IMG    = DATASET_ROOT / "val"

# Registrer (idempotent)
def safe_register(name, json_p, img_dir):
    try:
        register_coco_instances(name, {}, str(json_p), str(img_dir))
    except Exception:
        # allerede registreret i samme proces
        pass

safe_register("leaflet_train", TRAIN_JSON, TRAIN_IMG)
safe_register("leaflet_val",   VAL_JSON,   VAL_IMG)

print("DEBUG datasets:", DatasetCatalog.list())
try:
    print("DEBUG leaflet_train size:", len(DatasetCatalog.get("leaflet_train")))
    print("DEBUG leaflet_val   size:", len(DatasetCatalog.get("leaflet_val")))
except KeyError:
    pass


# ----------------- TRAINER -----------------
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # Brug den "gamle" signatur der matcher din detectron2-version
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )

    # DATA
    cfg.DATASETS.TRAIN = ("leaflet_train",)
    cfg.DATASETS.TEST  = ("leaflet_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # Hvis du vil beholde billeder uden annotationer, sæt til False (valgfrit):
    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # MODEL
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # SOLVER
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 2.5e-4
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = []               # ingen LR-decay steps
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000 # gem pth hver 2000 iter

    # EVAL + OUTPUT
    cfg.TEST.EVAL_PERIOD = 2000         # eval hver 2000 iter
    cfg.OUTPUT_DIR = "./output_detectron2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("🚀 Starting training...")

    trainer = MyTrainer(cfg)
    # resume=True så vi kan fortsætte automatisk hvis der findes checkpoints
    trainer.resume_or_load(resume=True)

    # BestCheckpointer for både bbox/AP og segm/AP
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    trainer.register_hooks([
        hooks.BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            checkpointer,
            val_metric="bbox/AP",
            mode="max",
            file_prefix="model_best_bbox",
        ),
        hooks.BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            checkpointer,
            val_metric="segm/AP",
            mode="max",
            file_prefix="model_best_segm",
        ),
    ])
    # Bemærk: DefaultTrainer opretter selv EvalHook baseret på cfg.TEST.EVAL_PERIOD,
    # og hook-rækkefølgen fungerer fint i praksis med ovenstående.

    trainer.train()

    # (Valgfrit) afsluttende eval på val
    evaluator = COCOEvaluator("leaflet_val", cfg, False,
                              output_dir=os.path.join(cfg.OUTPUT_DIR, "final_val_eval"))
    val_loader = build_detection_test_loader(cfg, "leaflet_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("📊 Final VAL results:", results)
    print(f"📁 Artefakter ligger i: {cfg.OUTPUT_DIR}")
    print("👉 Du får bl.a.:")
    print("   - model_best_bbox.pth  (bedst på bbox/AP)")
    print("   - model_best_segm.pth  (bedst på segm/AP)")
    print("   - model_final.pth      (sidste vægte)")
    print("   - model_00xxxx.pth     (periodiske checkpoints)")
    print("   - events.out.tfevents* (TensorBoard-logfiler)")


if __name__ == "__main__":
    main()

