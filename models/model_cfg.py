from detectron2.config import CfgNode as CN
from detectron2 import model_zoo
from detectron2.config import get_cfg


def get_cfg_mod(model_type="faster_rcnn_X_101_32x8d_FPN_3x", val=False):
    """
    Generates a Detectron2 configuration object based on the specified model type and evaluation mode.

    Parameters:
    - model_type (str): Name of the model to use.
    - val (bool): If True, set to validation mode.

    Returns:
    - cfg: Detectron2 configuration object.
    """
    cfg = get_cfg()
    MODEL = f"{model_type}.yaml"

    # Load model configurations from model zoo
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{MODEL}"))

    # Dataset specifications
    cfg.DATASETS.TRAIN = ("spine_train",)
    cfg.DATASETS.TEST = ("spine_val",) if val else ("spine_test",)

    # Input configurations
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.CROP = CN(
        {"ENABLED": True, "TYPE": "relative_range", "SIZE": [0.5, 0.5]})

    # Dataloader and Solver configurations
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.STEPS = (3000,)
    cfg.SOLVER.CLIP_GRADIENTS = CN(
        {"ENABLED": True, "CLIP_TYPE": "value", "CLIP_VALUE": 1.0, "NORM_TYPE": 2})

    # Model configurations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.PIXEL_MEAN = [87.693558721881,
                            5.365908382411402, 68.24405045361863]
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    # Other configurations
    cfg.CUDNN_BENCHMARK = True

    return cfg
