from Models.model_cfg import get_cfg_mod
from trainers.spineTrain import SpineTrainer
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#from google.colab.patches import cv2_imshow
import numpy as np
import os, json, cv2
import random, csv, sys

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
from Models.model_cfg import *
import argparse

def inference(dataset_dicts, cfg, outputDir, weightPath, val=False):
	cfg = get_cfg_mod(val=val)
	cfg.TEST.DETECTIONS_PER_IMAGE = 300  # used to be 100, but many dense images have close to 300 spines
	cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, outputDir)
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'weights', weightPath)  # path to the model we just trained

	##### NMS configs
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set a custom testing threshold
	cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3     # non-max suppression IoU threshold

	predictor = DefaultPredictor(cfg)

	outFile = os.path.join(cfg.OUTPUT_DIR, "predBoxes_josh_{}_test_cropped.txt".format(outputDir))

	with open(outFile, "w") as testOut:
		imgToBox = {"data" : {}}
		for d in dataset_dicts:
			im = cv2.imread(d["file_name"])
			print(d["file_name"])
			imgName = d["file_name"].split("/")[-1]
			outputs = predictor(im)
			boxes = outputs["instances"].get_fields()["pred_boxes"]
			imgToBox["data"][imgName] = []
			for box in boxes:
				imgToBox["data"][imgName].append(box.tolist())
		json.dump(imgToBox, testOut)






