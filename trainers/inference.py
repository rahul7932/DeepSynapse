import os
import cv2
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from Models.model_cfg import get_cfg_mod


def inference(dataset_dicts, cfg, output_dir, weight_path, val=False):
    cfg = get_cfg_mod(val=val)
    cfg.TEST.DETECTIONS_PER_IMAGE = 300  # Adjust as per your dataset needs
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_dir)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'weights', weight_path)

    # NMS configurations
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    predictor = DefaultPredictor(cfg)
    output_file = os.path.join(
        cfg.OUTPUT_DIR, f"predBoxes_josh_{output_dir}_test_cropped.txt")

    predictions = {"data": {}}
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        img_name = os.path.basename(d["file_name"])
        outputs = predictor(im)
        boxes = outputs["instances"].get_fields()["pred_boxes"].tensor.tolist()

        predictions["data"][img_name] = boxes

    with open(output_file, "w") as out:
        json.dump(predictions, out)
