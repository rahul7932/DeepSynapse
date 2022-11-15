import os
from utils.load_dataset import load_dataset

# TODO(vir): figure out why this cant be installed
import nni
from nni.utils import merge_parameter

from utils.get_params import get_params
from trainers import SpineTrainer


def trial(param_dict):
    dataDir = os.path.join(currDir, 'spineData', 'images2D_croppedBackbone')
    _, _, _ = load_dataset(dataDir, 'train')
    _, _, cfg = load_dataset(dataDir, 'val')

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'nni', 'weights')

    # params to tune for val accuracy

    cfg.SOLVER.BASE_LR = param_dict['lr']  # default is 0.001
    cfg.SOLVER.MOMENTUM = param_dict['momentum']   # default is 0.9
    cfg.SOLVER.MAX_ITER = 50000
    cfg.SOLVER.CHECKPOINT_PERIOD = 50000
    cfg.SOLVER.STEPS = (param_dict['gamma_step'],)

    cfg.TEST.EVAL_PERIOD = 50000
    cfg.TEST.EXPECTED_RESULTS = [['bbox', 'AP', 38.5, 0.2]]

    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = param_dict['roi_batch_size']
    # which layer of resnet to freeze up to (1-5), default is 2
    cfg.MODEL.BACKBONE.FREEZE_AT = param_dict['freeze_at']
    # loss function weighting, default is 1.0
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = param_dict['rpn_bbox_loss_weight']
    # loss function weighting, default is 1.0
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = param_dict['roi_bbox_loss_weight']
    # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param_dict['roi_score_thresh']
    # non-max suppression IoU threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = param_dict['roi_nms_thresh']

    # see if we can get intermediate results by augmenting default train loop in TrainerBase or DefaultTrainer
    trainer = SpineTrainer(cfg)
    results = trainer.train()

    print(f'info: config: {params_for_trial}')
    print(f'info: results: {results}')
    val_AP = results['bbox']['AP']

    # run full evaluation on trained model
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth') # path
    # predictor = DefaultPredictor(cfg)
    # evaluator = COCOEvaluator("spine_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # val_loader = build_detection_test_loader(cfg, "spine_val")
    # eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    nni.report_final_result(val_AP)


if __name__ == '__main__':
    try:
        # get parameters from tuner
        currDir = os.path.dirname(os.path.realpath(__file__))
        tuner_params = nni.get_next_parameter()
        params_for_trial = merge_parameter(get_params(), tuner_params)

        trial(params_for_trial)

    except Exception as exception:
        print(f'err: {exception}')