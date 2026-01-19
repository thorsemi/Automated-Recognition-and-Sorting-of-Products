


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_plots_report.py
Laver:
- PR-kurver (bbox & segm)
- F1 vs recall
- Confusion matrix (inkl. "error heat map")
- 10 eksempel-viz med predictions
- Træningskurver (loss m.m.) fra metrics.json
- t-SNE og UMAP af prædikerede maskers form (flattenede 32x32)
Kører på HPC uden GUI. Gemmer alt i output_detectron2/report_plots
"""

import os, json, math, random, itertools, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import cv2
import torch
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances

# ---------- PATHS ----------
ROOT = Path("/work3/s241614/projects/detectron2-scripts")
OUT = ROOT / "output_detectron2"
OUT_REPORT = OUT / "report_plots"
OUT_REPORT.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = Path("/work3/s241614/datasets/custom_dataset")
TEST_JSON = DATASET_ROOT / "annotations/instances_test.json"
TEST_IMG  = DATASET_ROOT / "test"

EVAL_DIR = OUT / "test_eval"
BBOX_RESULTS = EVAL_DIR / "bbox" / "coco_instances_results.json"
SEGM_RESULTS = EVAL_DIR / "segm" / "coco_instances_results.json"
METRICS_JSON = OUT / "metrics.json"

BEST_BBOX = OUT / "model_best_bbox.pth"
BEST_SEGM = OUT / "model_best_segm.pth"

# ---------- Utils ----------
def safe_read_jsonl_metrics(path):
    iters, losses = [], {"total_loss":[], "loss_cls":[], "loss_box_reg":[], "loss_mask":[]}
    if not path.exists():
        return iters, losses
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                continue
            if "iteration" in d and "total_loss" in d:
                iters.append(d["iteration"])
                for k in list(losses.keys()):
                    if k in d:
                        losses[k].append(float(d[k]))
                    else:
                        losses[k].append(np.nan)
    return iters, losses

def plot_training_curves():
    iters, losses = safe_read_jsonl_metrics(METRICS_JSON)
    if not iters:
        print("⚠️  Ingen metrics.json fundet – skipper træningsplots.")
        return
    # Enkelt-figurer (ikke subplots for at undgå for meget på én figur)
    for key in ["total_loss", "loss_box_reg", "loss_cls", "loss_mask"]:
        y = np.array(losses[key], dtype=float)
        x = np.array(iters, dtype=int)
        plt.figure()
        plt.plot(x, y, label=key)
        plt.xlabel("iteration"); plt.ylabel(key); plt.legend()
        plt.title(key)
        plt.tight_layout()
        plt.savefig(OUT_REPORT / f"train_{key}.png", dpi=200)
        plt.close()
    print("✅ Træningskurver gemt.")

def coco_pr_arrays(coco_gt_json, coco_dt_json, iou_type="bbox"):
    cocoGt = COCO(str(coco_gt_json))
    cocoDt = cocoGt.loadRes(str(coco_dt_json))
    ev = COCOeval(cocoGt, cocoDt, iouType=iou_type)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    # precision shape: [TxRxKxAxM]
    # recall shape:    [TxKxAxM]
    return ev

def plot_pr_and_f1(ev, tag):
    # Gennemsnit over IoU & kategorier, area=all (a=0), maxDet=100 (m=-1)
    precisions = ev.eval["precision"]  # T,R,K,A,M
    recalls    = ev.params.recThrs     # (R,)
    if precisions.size == 0:
        print(f"⚠️  Tom precision-array for {tag}")
        return
    # tag overall curve (mean over IoU & K, area idx 0, last maxDet)
    p = np.mean(precisions[:, :, :, 0, -1], axis=(0,2))  # -> (R,)
    r = recalls                                        # -> (R,)

    # PR
    plt.figure()
    plt.plot(r, p, label=f"PR-{tag}")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.legend()
    plt.title(f"Precision-Recall ({tag})")
    plt.tight_layout()
    plt.savefig(OUT_REPORT / f"pr_{tag}.png", dpi=200)
    plt.close()

    # F1 = 2PR/(P+R) pr recall-punkt
    f1 = (2*p*r) / np.clip(p + r, 1e-12, None)
    plt.figure()
    plt.plot(r, f1, label=f"F1-{tag}")
    plt.xlabel("recall"); plt.ylabel("F1"); plt.legend()
    plt.title(f"F1 vs recall ({tag})")
    plt.tight_layout()
    plt.savefig(OUT_REPORT / f"f1_{tag}.png", dpi=200)
    plt.close()
    print(f"✅ PR- og F1-plots gemt for {tag}.")

def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def build_confusion_matrix(gt_json, dt_json, iou_thr=0.5):
    gt = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    dt = json.loads(Path(dt_json).read_text(encoding="utf-8"))

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    cats = [cat_id_to_name[k] for k in sorted(cat_id_to_name)]
    K = len(cats)
    name_to_idx = {name:i for i,name in enumerate(cats)}
    BG = K  # background index
    C = np.zeros((K+1, K+1), dtype=int)  # rows: GT, cols: Pred; last index = background

    # index GT by image_id
    g_by_img = {}
    for a in gt["annotations"]:
        g_by_img.setdefault(a["image_id"], []).append(a)
    # index DT by image_id
    d_by_img = {}
    for a in dt:
        d_by_img.setdefault(a["image_id"], []).append(a)

    # match per image (greedy IoU ignoring class for confusion)
    for img_id, gts in g_by_img.items():
        dts = d_by_img.get(img_id, [])
        used_dt = set()
        # convert to boxes xyxy
        gt_boxes = []
        for g in gts:
            x,y,w,h = g["bbox"]
            gt_boxes.append([x, y, x+w, y+h])
        dt_boxes = []
        for d in dts:
            x,y,w,h = d["bbox"]
            dt_boxes.append([x, y, x+w, y+h])

        # For hver GT, find bedste DT over iou_thr
        for gi, g in enumerate(gts):
            gi_box = gt_boxes[gi]
            best_iou, best_di = 0.0, -1
            for di, d in enumerate(dts):
                if di in used_dt: 
                    continue
                iou = iou_xyxy(gi_box, dt_boxes[di])
                if iou > best_iou:
                    best_iou, best_di = iou, di
            gt_idx = name_to_idx[cat_id_to_name[g["category_id"]]]
            if best_iou >= iou_thr and best_di >= 0:
                used_dt.add(best_di)
                pred_name = cat_id_to_name.get(dts[best_di]["category_id"], None)
                if pred_name is None:  # should not happen
                    C[gt_idx, BG] += 1
                else:
                    pred_idx = name_to_idx[pred_name]
                    C[gt_idx, pred_idx] += 1
            else:
                # FN -> GT vs background
                C[gt_idx, BG] += 1

        # resterende dts er FP -> background vs pred class
        for di, d in enumerate(dts):
            if di in used_dt: 
                continue
            pred_name = cat_id_to_name.get(d["category_id"], None)
            pred_idx = name_to_idx[pred_name] if pred_name is not None else BG
            C[BG, pred_idx] += 1

    labels = cats + ["background"]
    return C, labels

def plot_confusion(C, labels, fname_png, normalize=True, title="Confusion"):
    M = C.astype(float)
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True) + 1e-9
        M = M / row_sums
    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Pred"); plt.ylabel("GT")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fname_png, dpi=250)
    plt.close()

def build_cfg_for_infer(weights_path, with_mask=True, score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def save_viz_examples(n=10, score_thresh=0.5):
    # brug segm-vægte for flotte maskeviz
    cfg = build_cfg_for_infer(BEST_SEGM, with_mask=True, score_thresh=score_thresh)
    # register test hvis ikke gjort
    if "leaflet_test" not in MetadataCatalog.list():
        register_coco_instances("leaflet_test", {}, str(TEST_JSON), str(TEST_IMG))
    md = MetadataCatalog.get("leaflet_test")
    predictor = DefaultPredictor(cfg)

    coco = COCO(str(TEST_JSON))
    img_infos = coco.loadImgs(coco.getImgIds())
    random.seed(42)
    random.shuffle(img_infos)
    os.makedirs(OUT_REPORT / "viz10", exist_ok=True)

    saved = 0
    for info in img_infos:
        p = TEST_IMG / info["file_name"]
        if not p.is_file():
            continue
        img = cv2.imread(str(p))[:, :, ::-1]  # BGR->RGB
        out = predictor(img)
        v = Visualizer(img, metadata=md, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        vis = v.draw_instance_predictions(out["instances"].to("cpu"))
        cv2.imwrite(str(OUT_REPORT / "viz10" / f"{Path(info['file_name']).stem}.jpg"),
                    vis.get_image()[:, :, ::-1])
        saved += 1
        if saved >= n:
            break
    print(f"✅ Gemte {saved} viz-billeder i {OUT_REPORT/'viz10'}.")

def tsne_umap_on_masks(max_instances=500, score_thresh=0.5):
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        print("⚠️  scikit-learn mangler til t-SNE. Kør: pip install scikit-learn")
        return
    try:
        import umap
    except Exception:
        umap = None  # valgfri

    cfg = build_cfg_for_infer(BEST_SEGM, with_mask=True, score_thresh=score_thresh)
    if "leaflet_test" not in MetadataCatalog.list():
        register_coco_instances("leaflet_test", {}, str(TEST_JSON), str(TEST_IMG))
    predictor = DefaultPredictor(cfg)
    coco = COCO(str(TEST_JSON))
    img_infos = coco.loadImgs(coco.getImgIds())
    feats, labels = [], []

    for info in img_infos:
        imgp = TEST_IMG / info["file_name"]
        if not imgp.is_file():
            continue
        img = cv2.imread(str(imgp))[:, :, ::-1]
        out = predictor(img)["instances"].to("cpu")
        if len(out) == 0: 
            continue
        keep = out.scores.numpy() >= score_thresh
        masks = out.pred_masks.numpy()[keep]  # [N,H,W]
        cls   = out.pred_classes.numpy()[keep]
        for m, c in zip(masks, cls):
            # normaliser mask til 32x32 for shape-embedding
            m32 = cv2.resize(m.astype(np.uint8), (32,32), interpolation=cv2.INTER_AREA).reshape(-1)
            feats.append(m32.astype(np.float32))
            labels.append(int(c))
            if len(feats) >= max_instances:
                break
        if len(feats) >= max_instances:
            break
    if len(feats) < 5:
        print("⚠️  For få mask-instanser til t-SNE/UMAP – skipper.")
        return

    X = np.stack(feats, axis=0)
    y = np.array(labels)

    # t-SNE
    ts = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=min(30, max(5, len(X)//10)))
    Z = ts.fit_transform(X)
    plt.figure()
    for k in np.unique(y):
        sel = y==k
        plt.scatter(Z[sel,0], Z[sel,1], s=8, label=str(k))
    plt.legend(title="class")
    plt.title("t-SNE (mask shape embedding)")
    plt.tight_layout()
    plt.savefig(OUT_REPORT / "tsne_masks.png", dpi=250)
    plt.close()

    # UMAP (hvis tilgængelig)
    if umap is not None:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        U = reducer.fit_transform(X)
        plt.figure()
        for k in np.unique(y):
            sel = y==k
            plt.scatter(U[sel,0], U[sel,1], s=8, label=str(k))
        plt.legend(title="class")
        plt.title("UMAP (mask shape embedding)")
        plt.tight_layout()
        plt.savefig(OUT_REPORT / "umap_masks.png", dpi=250)
        plt.close()
    print("✅ t-SNE/UMAP (masker) gemt.")

def main():
    # sanity
    for p in [TEST_JSON, TEST_IMG, BBOX_RESULTS, SEGM_RESULTS]:
        if not Path(p).exists():
            print(f"⚠️  Mangler fil/mappe: {p}")
    # 1) Træningskurver
    plot_training_curves()

    # 2) PR-kurver + F1
    ev_bbox = coco_pr_arrays(TEST_JSON, BBOX_RESULTS, "bbox")
    ev_segm = coco_pr_arrays(TEST_JSON, SEGM_RESULTS, "segm")
    plot_pr_and_f1(ev_bbox, "bbox")
    plot_pr_and_f1(ev_segm, "segm")

    # 3) Confusion matrix (bbox)
    C, labels = build_confusion_matrix(TEST_JSON, BBOX_RESULTS, iou_thr=0.5)
    np.savetxt(OUT_REPORT / "confusion_raw.csv", C, fmt="%d", delimiter=",")
    plot_confusion(C, labels, OUT_REPORT / "confusion_norm.png",
                   normalize=True, title="Confusion (IoU≥0.5, norm)")
    plot_confusion(C, labels, OUT_REPORT / "confusion_counts.png",
                   normalize=False, title="Confusion counts")
    print("✅ Confusion matrix gemt (PNG + CSV).")

    # 4) 10 eksempel-viz
    save_viz_examples(n=10, score_thresh=0.5)

    # 5) t-SNE / UMAP på maskers form
    tsne_umap_on_masks(max_instances=500, score_thresh=0.5)

    print(f"📁 Output: {OUT_REPORT}")

if __name__ == "__main__":
    main()
