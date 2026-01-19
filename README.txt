All external digital material is organised in the Appendix folder, structured into the following subfolders:

OpenCV
Contains the complete classical computer-vision pipeline:

Gray/ — Grayscale-based detection and front/back classification scripts

RGB/ — Colour-based (HSV/RGB) implementations and experiments

Eval/ — Evaluation notebooks, plots, and exported results for the OpenCV models

YOLO
Contains all material related to the YOLO-based real-time detection models:

nano/, small/, medium/ — Trained YOLOv11 weights and configuration files for the three network sizes

Script/ — Training and inference scripts used in the project

test_evaluation/ — Evaluation outputs (metrics, plots, and example predictions)

Detectron2
Contains the Detectron2-based instance-segmentation and analysis pipeline:

Train/ — Training configuration, checkpoints, and logs

Test/ — Result images and evaluation exports (PR curves, confusion matrices, etc.)

Script/ — Python scripts for inference, heatmap generation, and lifecycle analysis

Line Crossing Detection
Contains the implementation of the virtual line-crossing detection logic:

Line Crossing Detection.py — Python script used to generate digital trigger signals for the PLC based on virtual line crossings

Cognex
Contains the configuration for the benchmark industrial vision system:

Model.jobx — Cognex In-Sight job file used to configure and run the reference vision application