import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class DetectronModel:
    """Wrapper class for Detectron2 model loading and inference."""

    def __init__(self, weight_path, num_classes=7, threshold=0.5):
        print(f"Loading Detectron2 model from: {weight_path}")
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(cfg)
        self.metadata = (
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            if len(cfg.DATASETS.TRAIN)
            else MetadataCatalog.get("__unused__")
        )
        # Brug et nyt metadata-objekt i stedet for COCO
        self.metadata = MetadataCatalog.get("custom_classes")
        self.metadata.set(thing_classes=[
            "back", "front", "robot", "2_overlap", "3_overlap", "heavy_overlap", "partiel"
        ])


    def predict(self, frame, return_boxes=False):
        """Runs Detectron2 inference and returns annotated frame (and optionally box centers)."""
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")

        # Visualiserer billedet
        vis = Visualizer(frame[:, :, ::-1], metadata=self.metadata, scale=1.0)
        vis_output = vis.draw_instance_predictions(instances)
        annotated = vis_output.get_image()[:, :, ::-1]

        # Beregn midtpunkter (cx, cy)
        centers = []
        if return_boxes:
            for box in instances.pred_boxes:
                x1, y1, x2, y2 = box.tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))

        return (annotated, centers) if return_boxes else annotated
