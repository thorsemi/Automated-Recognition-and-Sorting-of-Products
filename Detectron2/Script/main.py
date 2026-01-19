import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

model_path = r"C:\Users\emilt\Desktop\Appendix\Detectron2\Train\output_detectron2\model_best_segm.pth"
img_path = r"C:\Users\emilt\Desktop\DTU\Semester 2\Specialkursus - PJM Kursus\yolo_dataset\images\val\enkle_back_003485.png"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7        
cfg.MODEL.DEVICE = "cpu"                   

predictor = DefaultPredictor(cfg)

img = cv2.imread(img_path)
outputs = predictor(img)

custom_metadata = MetadataCatalog.get("__unused")
custom_metadata.thing_classes = [
    "Back",
    "Front",
    "Robot",
    "2_overlap",
    "3_overlap",
    "Heavy_overlap",
    "Partial",
]

v = Visualizer(img[:, :, ::-1], custom_metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imshow("Detectron2 Resultat", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
