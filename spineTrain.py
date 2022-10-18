#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepSynapse/DeepSpine Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch
import nni
from nni.utils import merge_parameter
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from spine_utils import *

def build_spine_train_aug(cfg):
    #print(cfg)
    augs = [
        #T.ResizeShortestEdge(
        #            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        #        )
    ]
    #if cfg.INPUT.CROP.ENABLED:
    #    augs.append(
    #        T.RandomCrop(
    #            cfg.INPUT.CROP.TYPE,
    #            cfg.INPUT.CROP.SIZE
    #        )
    #    )
    #augs.append(T.RandomFlip())
    #augs.append(T.RandomRotation([0.0, 180.0]))
    #augs.append(T.RandomSaturation(0.3, 2))
    #augs.append(T.RandomContrast(0.3, 2))
    #augs.append(T.RandomBrightness(0.3, 2))
    return augs

class SpineTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return COCOEvaluator("spine_val", cfg, False, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_spine_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

        
## can do this in a better, less redundant way
## this is just first attempt at merging detectron configs with nni requirements
def get_params():
    param_dict = {} 
    param_dict['roi_batch_size'] = 32
    param_dict['lr'] = 0.001
    param_dict['momentum'] = 0.9
    param_dict['gamma_step'] = 10
    param_dict['gamma'] = 0.1
    param_dict['freeze_at'] = 1
    param_dict['rpn_bbox_loss_weight'] = 1.0
    param_dict['roi_bbox_loss_weight'] = 1.0
    param_dict['roi_score_thresh'] = 0.5
    param_dict['roi_nms_thresh'] = 0.3
    return param_dict


def trial(param_dict):
    dataDir = os.path.join(currDir, 'spineData', 'images2D_croppedBackbone')
    dataset_dicts_train, spine_metadata_train, _ = load_dataset(dataDir, 'train')
    dataset_dicts_val, spine_metadata_val, cfg = load_dataset(dataDir, 'val')

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'nni', 'weights')

    # params to tune for val accuracy

    cfg.SOLVER.BASE_LR = param_dict['lr']  # default is 0.001

    cfg.SOLVER.MOMENTUM = param_dict['momentum']   # default is 0.9

    cfg.SOLVER.MAX_ITER = 50000

    cfg.SOLVER.CHECKPOINT_PERIOD = 50000

    cfg.TEST.EVAL_PERIOD = 50000

    cfg.TEST.EXPECTED_RESULTS = [['bbox', 'AP', 38.5, 0.2]]

    cfg.SOLVER.STEPS = (param_dict['gamma_step'],)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = param_dict['roi_batch_size']  # faster, and good enough for this toy dataset (default: 512)

    cfg.MODEL.BACKBONE.FREEZE_AT = param_dict['freeze_at']   # which layer of resnet to freeze up to (1-5), default is 2    

    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = param_dict['rpn_bbox_loss_weight'] # loss function weighting, default is 1.0

    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = param_dict['roi_bbox_loss_weight'] # loss function weighting, default is 1.0

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param_dict['roi_score_thresh']   # set a custom testing threshold

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = param_dict['roi_nms_thresh']     # non-max suppression IoU threshold

    ## see if we can get intermediate results by augmenting default train loop in TrainerBase or DefaultTrainer
    trainer = SpineTrainer(cfg) 
    results = trainer.train() 
    print('Results: ')
    print(results)
    val_AP = results['bbox']['AP']
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')  # path to the model we just trained
    #predictor = DefaultPredictor(cfg)
    #evaluator = COCOEvaluator("spine_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    #val_loader = build_detection_test_loader(cfg, "spine_val")
    #eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    #
    #val_AP = eval_results['bbox']['AP']
    #
    # report final result
    nni.report_final_result(val_AP)
    #logger.debug('Final result is %g', results)
    #logger.debug('Send final result done.')

if __name__ == '__main__':
    try:
        # get parameters from tuner
        currDir = os.path.dirname(os.path.realpath(__file__))
        tuner_params = nni.get_next_parameter()
        #logger.debug(tuner_params)
        params_for_trial = merge_parameter(get_params(), tuner_params)
        print(params_for_trial)
        trial(params_for_trial)
    except Exception as exception:
        print(exception)
        raise