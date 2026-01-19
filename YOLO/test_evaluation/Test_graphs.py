from ultralytics import YOLO
import os

models = {
    "Nano":   r"C:\Users\emilt\Desktop\YOLO\yolo_nano\yolo_nano_min_20251029_1813\runs\train\y12n_prod_tune\weights\best.pt",
    "Small":  r"C:\Users\emilt\Desktop\YOLO\yolo_small\y11s_prod_bundle\y11s_prod_bundle\best.pt",
    "Medium": r"C:\Users\emilt\Desktop\YOLO\yolo_medium\Run2\y11m_medium_only_20251030\medium_only_20251030\weights\y11m_prod_tune_best.pt",
}

tests = {
    "Skygge": r"C:\Users\emilt\Desktop\data_black.yaml",
    "Fabrikslys": r"C:\Users\emilt\Desktop\data_fabrik.yaml",
    "Lyst": r"C:\Users\emilt\Desktop\data_lyst.yaml",
}

save_dir = r"C:\Users\emilt\Desktop\YOLO\Test_results"

for cond_name, data_yaml in tests.items():
    for model_name, model_path in models.items():
        print(f"\nTesting {model_name} on {cond_name} (TEST)…")
        model = YOLO(model_path)
        r = model.val(
            data=data_yaml,
            split="test",       
            imgsz=896,
            conf=0.25,
            iou=0.5,
            plots=True,         
            project=save_dir,
            name=f"{model_name}_{cond_name}",
            verbose=False
        )
        print(model_name, cond_name, r.results_dict["metrics/recall(B)"])
