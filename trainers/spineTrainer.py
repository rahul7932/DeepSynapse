import os
import torch
import nni
from detectron2.data.transforms import DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from Models.model_cfg import *
from trainers.augmentation import build_spine_train_aug


class SpineTrainer(DefaultTrainer):
    """
    Custom training class for spine images based on Detectron2's DefaultTrainer.
    """

    @classmethod
    def build_evaluator(cls, cfg):
        """
        Create an evaluator for the spine validation dataset.
        """
        return COCOEvaluator("spine_val", cfg, False, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build a data loader for training using custom augmentations.
        """
        mapper = DatasetMapper(
            cfg, is_train=True, augmentations=build_spine_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
