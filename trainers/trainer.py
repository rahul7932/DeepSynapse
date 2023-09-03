import os
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from Models.model_cfg import get_cfg_mod
from trainers.spineTrain import SpineTrainer

setup_logger()


def train(dataset_dicts, spine_metadata, cfg, outputDir, toResume=False, weightPath='model_0044999.pth', vis=False):
    """
    Train a model on a given dataset with Detectron2.

    Parameters:
    - dataset_dicts (list): List of dataset dictionaries.
    - spine_metadata (dict): Metadata of the spine dataset.
    - cfg (object): Configuration object for Detectron2.
    - outputDir (str): Directory for outputting trained weights.
    - toResume (bool): If true, resume training.
    - weightPath (str): Path to model weights if not training.
    - vis (bool): If true, visualize bounding boxes.

    Returns:
    None.
    """

    # VISUALIZATIONS FOR BOUNDING BOXES
    if vis:
        for d in dataset_dicts[:5]:
            print(f"Visualizing GT data for {d['file_name']}")
            img = cv2.imread(d["file_name"], cv2.IMREAD_GRAYSCALE)
            print(f"Contains {len(d['annotations'])} boxes")
            visualizer = Visualizer(
                img[:, :], metadata=spine_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            plt.imshow(out.get_image()[:, :, ::-1])
            plt.show()

    # Configuration settings
    cfg = get_cfg_mod()
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, outputDir, 'weights')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize and set up trainer
    trainer = SpineTrainer(cfg)
    if not train:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightPath)
    trainer.resume_or_load(resume=toResume)
    print(f'Resume? : {toResume}')

    # Set GPU fan speed for temperature control
    os.system("nvidia-settings -a '[gpu:0]/GPUFanControlState=1'")
    os.system("nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=80'")
    os.system("nvidia-settings -a '[fan:1]/GPUTargetFanSpeed=80'")

    # Train
    trainer.train()

    # Reset GPU fan speed
    os.system("nvidia-settings -a '[gpu:0]/GPUFanControlState=0'")


# Reminder:
# if training is manually stopped make sure to run the following command in terminal:
# nvidia-settings -a '[gpu:0]/GPUFanControlState=0'
