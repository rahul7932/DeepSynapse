import os

import cv2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

from models.model_cfg import *
from models.model_cfg import get_cfg_mod
from trainer import SpineTrainer
setup_logger()


def train(dataset_dicts, spine_metadata, cfg, outputDir, toResume=False, weightPath='model_0044999.pth', vis=False):
    # VISUALIZATIONS FOR BOUNDING BOXES
    if vis:
        # generating visualizations for the first 5 images in the dataset
        for d in dataset_dicts[:5]:
            print(f"info: visualizing GT data for {d['file_name']}")
            print(f"info: contains {len(d['annotations'])} boxes")

            img = cv2.imread(d["file_name"], cv2.IMREAD_GRAYSCALE)
            visualizer = Visualizer(img[:, :], metadata=spine_metadata, scale=1)

            out = visualizer.draw_dataset_dict(d)
            plt.imshow(out.get_image()[:, :, ::-1])
            plt.show()

    cfg = get_cfg_mod()
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, outputDir, 'weights')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not train:
        # if we aren't training, pull the weights from the weight path
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightPath)

    trainer = SpineTrainer(cfg)
    trainer.resume_or_load(resume=toResume)
    print(f'info: resume? {str(toResume)}')

    # manually set fan speed to high before training for temp control
    os.system("nvidia-settings -a '[gpu:0]/GPUFanControlState=1'")
    os.system("nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=80'")
    os.system("nvidia-settings -a '[fan:1]/GPUTargetFanSpeed=80'")

    trainer.train()

    # if training is manually stopped make sure to run the following command in terminal:
    # nvidia-settings -a '[gpu:0]/GPUFanControlState=0'
    os.system("nvidia-settings -a '[gpu:0]/GPUFanControlState=0'")