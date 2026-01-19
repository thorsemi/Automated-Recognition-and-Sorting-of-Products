#!/usr/bin/env python3
import os, json, re, numpy as np, torch
from pathlib import Path
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog

# ---- NumPy bool shim (undgår np.bool warnings) ----
if not hasattr(np, "bool"): np.bool = bool

# ---- DATA PATHS ----
ROOT       = Path("/work3/s241614/datasets/custom_dataset")
TRAIN_JSON = ROOT/"annotations/instances_train.json"
VAL_JSON   = ROOT/"annotations/instances_val.json"
TEST_JSON  = ROOT/"annotations/instances_test.json"
TRAIN_IMG  = ROOT/"train"
VAL_IMG    = ROOT/"val"
TEST_IMG   = ROOT/"test"

OUTDIR     = Path("./output_detectron2/test_eval")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- Klassernes RÆKKEFØLGE fra træning (vigtig!) ----
TRAIN_CLASS_ORDER = ["back","front","robot","2_overlap","3_overlap","heavy_overlap","partial"]

def fix_test_json_paths_and_danish_letters(json_path: Path, img_root: Path):
    """Ret evt. 'test/'-prefix i file_name + æ/ø/å der er blevet til Ã¦/Ã¸/Ã¥."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = False

    for im in data.get("images", []):
        fn = im.get("file_name", "")
        # fjern 'test/' prefix hvis den er der
        if fn.startswith("test/"):
            im["file_name"] = fn.split("test/",1)[1]
            changed = True
        # ret forkert encoding af æøå
        fixed = (im["file_name"]
                 .replace("trÃ¦","træ")
                 .replace("Ã¦","æ").replace("Ã¸","ø").replace("Ã¥","å")
                 .replace("Ã†","Æ").replace("Ã˜","Ø").replace("Ã…","Å"))
        if fixed != im["file_name"]:
            im["file_name"] = fixed
            changed = True

    # sørg for info/licences felter (COCO kompatibilitet)
    if "info" not in data:     data["info"] = {"description":"patched","version":"1.0"}; changed = True
    if "licenses" not in data: data["licenses"] = []; changed = True

    if changed:
        backup = json_path.with_suffix(json_path.suffix + ".bak_before_fix")
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Patchet filnavne/metadata i {json_path} (backup: {backup})")
    return data

def enforce_test_category_mapping_by_name(test_json_path: Path, dataset_name: str):
    """
    Sørg for at evaluatorens mapping for leaflet_test matcher TRAIN_CLASS_ORDER ved at mapppe ved 'name'.
    """
    meta = MetadataCatalog.get(dataset_name)
    with open(test_json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    name2id = {c["name"]: c["id"] for c in d.get("categories", [])}
    missing = [n for n in TRAIN_CLASS_ORDER if n not in name2id]
    if missing:
        print("⚠️  Mangler kategorier i test JSON:", missing)
        print("    Kan give AP=0. Tjek categories i", test_json_path)
    # byg dataset_id -> contiguous_id mapping efter træningsrækkefølge
    thing_dataset_id_to_contiguous_id = {}
    for contig, name in enumerate(TRAIN_CLASS_ORDER):
        if name in name2id:
            thing_dataset_id_to_contiguous_id[name2id[name]] = contig

    # Sæt metadata så COCOEvaluator bruger denne mapping
    MetadataCatalog.get(dataset_name).set(
        thing_classes=TRAIN_CLASS_ORDER,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id
    )
    # (COCOEvaluator danner selv reverse mapping internt)
    print("✅ Tvunget test-mapping efter træningsrækkefølge.")

def build_cfg(weights_path: str, num_classes: int = 7, score_thresh: float = 0.30):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MASK_FORMAT = "polygon"
    return cfg

def evaluate_once(cfg, dataset_name, out_dir, tag="bbox"):
    os.makedirs(out_dir, exist_ok=True)
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=out_dir)
    loader = build_detection_test_loader(cfg, dataset_name)

    # Byg model + load weights
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    results = inference_on_dataset(model, loader, evaluator)
    out_json = os.path.join(out_dir, f"results_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Gemte resultater: {out_json}")
    return results

def sanity_predict_and_save(cfg, n=5, outdir=OUTDIR/"viz"):
    """Gem et par eksempelbilleder med forudsigelser (hurtig sanity-check)."""
    outdir.mkdir(parents=True, exist_ok=True)
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
    import cv2, random

    predictor = DefaultPredictor(cfg)
    meta = MetadataCatalog.get("leaflet_test")
    # hent første n billeder fra dataset
    ds = DatasetCatalog.get("leaflet_test")
    picks = ds[:n]
    for d in picks:
        im = cv2.imread(d["file_name"])
        out = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=meta, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(out["instances"].to("cpu"))
        out_path = outdir / (Path(d["file_name"]).stem + "_pred.jpg")
        cv2.imwrite(str(out_path), v.get_image()[:, :, ::-1])
    print(f"🖼  Gemte {len(picks)} viz-billeder i {outdir}")

def main():
    # 1) Patch JSON så paths matcher disk
    test_data = fix_test_json_paths_and_danish_letters(TEST_JSON, TEST_IMG)

    # 2) Registrer datasets (train kun for konsistens/metadata)
    if "leaflet_train" in DatasetCatalog.list():
        DatasetCatalog.remove("leaflet_train")
    if "leaflet_val" in DatasetCatalog.list():
        DatasetCatalog.remove("leaflet_val")
    if "leaflet_test" in DatasetCatalog.list():
        DatasetCatalog.remove("leaflet_test")

    register_coco_instances("leaflet_train", {}, str(TRAIN_JSON), str(TRAIN_IMG))
    register_coco_instances("leaflet_val",   {}, str(VAL_JSON),   str(VAL_IMG))
    register_coco_instances("leaflet_test",  {}, str(TEST_JSON),  str(TEST_IMG))

    # 3) Tving test-kategori-mapping efter træningsrækkefølge (kritisk for AP)
    enforce_test_category_mapping_by_name(TEST_JSON, "leaflet_test")

    # 4) Hurtig kontrol: findes alle filer?
    missing = []
    for im in test_data.get("images", []):
        if not (TEST_IMG / im["file_name"]).is_file():
            missing.append(im["file_name"])
    print(f"✅ Antal billeder i JSON: {len(test_data.get('images', []))}")
    print(f"❌ Mangler på disk: {len(missing)}")
    if missing:
        print("   Eksempler:", missing[:10])

    # 5) Evaluer BBOX
    bbox_w = "output_detectron2/model_best_bbox.pth"
    print(f"▶️ Evaluerer BBOX med weights: {bbox_w}")
    cfg_bbox = build_cfg(bbox_w, num_classes=len(TRAIN_CLASS_ORDER), score_thresh=0.30)
    res_bbox = evaluate_once(cfg_bbox, "leaflet_test", str(OUTDIR), tag="bbox")

    # 6) Evaluer SEGM
    segm_w = "output_detectron2/model_best_segm.pth"
    print(f"▶️ Evaluerer SEGM med weights: {segm_w}")
    cfg_segm = build_cfg(segm_w, num_classes=len(TRAIN_CLASS_ORDER), score_thresh=0.30)
    res_segm = evaluate_once(cfg_segm, "leaflet_test", str(OUTDIR), tag="segm")

    # 7) Gem nogle sanity-viz
    sanity_predict_and_save(cfg_bbox, n=5)

    print("📁 Outputdir:", OUTDIR)
    print("📌 Filer du får:\n"
          "   - results_bbox.json / results_segm.json (COCO metrics)\n"
          "   - coco_instances_results.json + instances_predictions.pth i underfoldere\n"
          "   - viz/*.jpg med predictions\n"
          )

if __name__ == "__main__":
    main()
