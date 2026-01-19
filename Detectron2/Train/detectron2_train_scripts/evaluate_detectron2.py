#!/usr/bin/env python3
import os, json, random, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.manifold import TSNE

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Boxes
from detectron2.structures.boxes import pairwise_iou

OUT = Path("output_detectron2")
SAMPLES_DIR = Path("samples"); SAMPLES_DIR.mkdir(exist_ok=True)
random.seed(42); np.random.seed(42)

# --- dataset registration (samme paths som i træning)

from pathlib import Path
DATASET_ROOT = Path("/work3/s241614/datasets/custom_dataset")
TRAIN_JSON = DATASET_ROOT/"annotations/instances_train.json"
VAL_JSON   = DATASET_ROOT/"annotations/instances_val.json"
TRAIN_IMG  = DATASET_ROOT/"train"
VAL_IMG    = DATASET_ROOT/"val"
TEST_JSON  = DATASET_ROOT/"annotations/instances_test.json"
TEST_IMG   = DATASET_ROOT/"test"

register_coco_instances("leaflet_train", {}, str(TRAIN_JSON), str(TRAIN_IMG))
register_coco_instances("leaflet_val",   {}, str(VAL_JSON),   str(VAL_IMG))
if TEST_JSON.exists() and TEST_IMG.exists():
    register_coco_instances("leaflet_test",  {}, str(TEST_JSON),  str(TEST_IMG))


meta = MetadataCatalog.get("leaflet_train")
cats = meta.get("thing_classes", None)
if not cats:
    # hent fra JSON hvis ikke sat
    with open("annotations/annotations_train.json","r") as f:
        cats = [c["name"] for c in json.load(f)["categories"]]
    MetadataCatalog.get("leaflet_train").thing_classes = cats
    MetadataCatalog.get("leaflet_val").thing_classes = cats
    MetadataCatalog.get("leaflet_test").thing_classes = cats
num_classes = len(cats)

# --- helper
def read_metrics(path=OUT/"metrics.json"):
    it, loss, lr, ap = [], [], [], []
    if not path.exists(): return it, loss, lr, ap
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if "iteration" in d:
                if "total_loss" in d:      loss.append((d["iteration"], d["total_loss"]))
                if "lr" in d:               lr.append((d["iteration"], d["lr"]))
            # AP logges kun ved eval-steps
            if "bbox/AP" in d:
                ap.append((d["iteration"], d["bbox/AP"]))
            if "iteration" in d:
                it.append(d["iteration"])
    return it, loss, lr, ap

def xywh_to_xyxy(b):
    x,y,w,h = b
    return [x, y, x+w, y+h]

# --- 1) Loss curve
_, loss, lr, ap = read_metrics()
def plot_series(pairs, title, ylabel, out):
    if not pairs: return
    xs, ys = zip(*pairs)
    plt.figure(); plt.plot(xs, ys)
    plt.title(title); plt.xlabel("iteration"); plt.ylabel(ylabel); plt.grid(True)
    plt.tight_layout(); plt.savefig(out); plt.close()

plot_series(loss, "Total loss", "loss", "loss_curve.png")

# --- 2) LR schedule
plot_series(lr, "Learning rate", "lr", "lr_schedule.png")

# --- 3) mAP progression
plot_series(ap, "Validation mAP (bbox/AP)", "AP", "map_progression.png")

# --- 4) Class distribution (train)
with open("annotations/annotations_train.json","r") as f:
    tr = json.load(f)
catmap = {c["id"]: c["name"] for c in tr["categories"]}
counts = Counter([a["category_id"] for a in tr["annotations"]])
plt.figure()
plt.bar([catmap[k] for k in counts.keys()], [counts[k] for k in counts.keys()])
plt.xticks(rotation=45, ha="right"); plt.ylabel("instances")
plt.title("Class distribution (train)")
plt.tight_layout(); plt.savefig("class_distribution.png"); plt.close()

# --- Predictor (load best/final checkpoint)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
final_pth = OUT/"model_final.pth"
if final_pth.exists():
    cfg.MODEL.WEIGHTS = str(final_pth)
else:
    # fallback til sidste checkpoint hvis final ikke findes
    last = sorted(OUT.glob("model_*.pth"))[-1]
    cfg.MODEL.WEIGHTS = str(last)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# --- 5) Sample predictions (5 tilfældige testbilleder)
ds = DatasetCatalog.get("leaflet_test")
for i, rec in enumerate(random.sample(ds, min(5, len(ds)))):
    img = cv2.imread(rec["file_name"])
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("leaflet_test"),
                   scale=0.7, instance_mode=ColorMode.IMAGE)
    out = predictor(img)
    v = v.draw_instance_predictions(out["instances"].to("cpu"))
    cv2.imwrite(str(SAMPLES_DIR/f"sample_{i+1}.jpg"), v.get_image()[:, :, ::-1])

# --- 6) Confusion matrix + F1 curve (IoU=0.5, micro-averaged)
def compute_confusion_and_f1(dataset, iou_thr=0.5, thresholds=np.linspace(0.05,0.95,19)):
    cm = np.zeros((num_classes, num_classes+1), dtype=int)  # last column = missed (FN -> bg)
    # for F1 over thresholds
    tps = defaultdict(int); fps = defaultdict(int); fns = defaultdict(int)

    for rec in dataset:
        img = cv2.imread(rec["file_name"])
        ann = rec.get("annotations", [])
        gt_boxes = np.array([xywh_to_xyxy(a["bbox"]) for a in ann], dtype=np.float32) if ann else np.zeros((0,4), np.float32)
        gt_cls   = np.array([a["category_id"]-1 for a in ann], dtype=int) if ann else np.zeros((0,), int)

        out = predictor(img)["instances"].to("cpu")
        pb = out.pred_boxes.tensor.numpy() if len(out)>0 else np.zeros((0,4), np.float32)
        ps = out.scores.numpy() if len(out)>0 else np.zeros((0,), np.float32)
        pc = out.pred_classes.numpy() if len(out)>0 else np.zeros((0,), int)

        # match for confusion
        if len(gt_boxes)==0 and len(pb)>0:
            for c in pc: cm[:, c]  # nothing to do; pure FP handled in F1 loop
        if len(gt_boxes)>0:
            ious = pairwise_iou(Boxes(torch.tensor(gt_boxes)), Boxes(torch.tensor(pb))).numpy() if len(pb)>0 else np.zeros((len(gt_boxes),0))
            matched_pred = set()
            for gi in range(len(gt_boxes)):
                # find best pred for this gt
                if ious.shape[1]==0 or ious[gi].max()<iou_thr:
                    cm[gt_cls[gi], -1] += 1  # miss
                else:
                    pj = int(np.argmax(ious[gi]))
                    if pj in matched_pred:
                        cm[gt_cls[gi], -1] += 1
                    else:
                        cm[gt_cls[gi], pc[pj]] += 1
                        matched_pred.add(pj)

        # F1 over thresholds (micro)
        for thr in thresholds:
            keep = ps >= thr
            pb_keep, pc_keep = pb[keep], pc[keep]
            # recompute matches for this threshold
            if len(gt_boxes)==0:
                fps[thr] += len(pb_keep); continue
            if len(pb_keep)==0:
                fns[thr] += len(gt_boxes); continue
            ious_thr = pairwise_iou(Boxes(torch.tensor(gt_boxes)), Boxes(torch.tensor(pb_keep))).numpy()
            used_p = set(); used_g = set()
            for gi in range(len(gt_boxes)):
                pj = int(np.argmax(ious_thr[gi]))
                if ious_thr[gi, pj] >= iou_thr and pj not in used_p:
                    tps[thr] += 1
                    used_p.add(pj); used_g.add(gi)
                else:
                    fns[thr] += 1
            fps[thr] += (len(pb_keep) - len(used_p))
    # F1 curve
    xs, f1s = [], []
    for thr in thresholds:
        tp, fp, fn = tps[thr], fps[thr], fns[thr]
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec+rec+1e-9)
        xs.append(thr); f1s.append(f1)
    return cm, (xs, f1s)

cm, (xs, f1s) = compute_confusion_and_f1(ds)

# Confusion matrix plot
plt.figure(figsize=(1.2*num_classes, 1.0*num_classes))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                 xticklabels=cats+["(miss)"], yticklabels=cats)
ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title("Confusion matrix (IoU=0.5)")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.close()

# F1 curve
plt.figure()
plt.plot(xs, f1s)
plt.xlabel("score threshold"); plt.ylabel("F1 (micro)"); plt.title("F1 vs threshold (IoU=0.5)")
plt.grid(True); plt.tight_layout(); plt.savefig("f1_curve.png"); plt.close()

# --- 7) t-SNE features (ROI features for top detections på ~50 billeder)
def tsne_features(dataset, max_images=50, max_dets_per_img=20):
    feats, labels = [], []
    model = predictor.model
    model.eval()
    with torch.no_grad():
        for rec in random.sample(dataset, min(len(dataset), max_images)):
            img = cv2.imread(rec["file_name"])
            h,w = img.shape[:2]
            inputs = [{"image": torch.as_tensor(img.astype("float32").transpose(2,0,1)), "height": h, "width": w}]
            # backbone features
            images = model.preprocess_image(inputs)
            feats_map = model.backbone(images.tensor)
            # run RPN+ROI quickly via predictor to get boxes/classes
            out = predictor(img)["instances"].to("cpu")
            if len(out)==0: continue
            # pool ROI features at predicted boxes
            in_feat_names = model.roi_heads.in_features
            feat_list = [feats_map[f] for f in in_feat_names]
            boxes = [Boxes(out.pred_boxes.tensor.to(feat_list[0].device))]
            pooled = model.roi_heads.box_pooler(feat_list, boxes)
            box_feats = model.roi_heads.box_head(pooled)  # [N, D]
            box_feats = box_feats[:max_dets_per_img].detach().cpu().numpy()
            cls = out.pred_classes.numpy()[:box_feats.shape[0]]
            feats.append(box_feats); labels.append(cls)
    if not feats: return None, None
    X = np.vstack(feats); y = np.concatenate(labels)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, n_iter=1000)
    Z = tsne.fit_transform(X)
    return Z, y

Z, y = tsne_features(ds)
if Z is not None:
    plt.figure()
    for c in range(num_classes):
        m = y==c
        if m.sum(): plt.scatter(Z[m,0], Z[m,1], s=8, label=cats[c], alpha=0.7)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.title("t-SNE of ROI features")
    plt.tight_layout(); plt.savefig("tsne_features.png"); plt.close()

print("✅ Plots skrevet:")
print(" - samples/sample_*.jpg")
print(" - confusion_matrix.png")
print(" - tsne_features.png")
print(" - map_progression.png")
print(" - f1_curve.png")
print(" - class_distribution.png")
print(" - lr_schedule.png")
print(" - loss_curve.png")
