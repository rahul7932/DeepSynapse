from detectron2.config import CfgNode as CN
from detectron2 import model_zoo
from detectron2.config import get_cfg

def get_cfg_mod(model_type="faster_rcnn_X_101_32x8d_FPN_3x", val=False):
	# CONFIGS
	# detectron config object
	cfg = get_cfg()
	MODEL = model_type + ".yaml"
	print(MODEL)

	# fRCNN50 = "faster_rcnn_R_50_FPN_1x.yaml"
	# fRCNN101 = "faster_rcnn_R_101_DC5_3x.yaml"
	# retinaNet101 = "retinanet_R_101_FPN_3x.yaml"
	# fRCNNX101 = "faster_rcnn_X_101_32x8d_FPN_3x.yaml"

	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/" + MODEL))
	cfg.INPUT.FORMAT = "BGR"
	cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
	cfg.INPUT.MIN_SIZE_TEST = 1024
	cfg.INPUT.MAX_SIZE_TRAIN = 1024
	cfg.INPUT.MAX_SIZE_TEST = 1024

	#Enable cropping augmentations during training
	cfg.INPUT.CROP = CN({"ENABLED": True})
	cfg.INPUT.CROP.TYPE = "relative_range"
	cfg.INPUT.CROP.SIZE = [0.5, 0.5]

	#cfg.INPUT.FORMAT = "L" 
	# default is "BGR", using "L" for 8-bit black and white spine images
	
	cfg.DATASETS.TRAIN = ("spine_train",)

	if not val:
		cfg.DATASETS.TEST = ("spine_test",)
	else:
		cfg.DATASETS.TEST = ("spine_val",)

	cfg.DATALOADER.NUM_WORKERS = 1
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.SOLVER.BASE_LR = 0.0025  # default is 0.001
	cfg.SOLVER.MOMENTUM = 0.9	# default is 0.9
	cfg.SOLVER.MAX_ITER = 10000 #150000
	cfg.SOLVER.CHECKPOINT_PERIOD = 500 #10000
	cfg.SOLVER.STEPS = (3000,)
	cfg.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True}) # gradient clipping is set to False by default
	cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
	# Maximum absolute value used for clipping gradients
	cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
	# Floating point number p for L-p norm to be used with the "norm"
	# gradient clipping type; for L-inf, please specify .inf
	cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # faster, and good enough for this toy dataset (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (spine)
	cfg.MODEL.BACKBONE.FREEZE_AT = 0   # which layer of resnet to freeze up to (1-5), default is 2    

	# Calculated dataset info for better training results
	# real mean: b 87.693558721881
	# real mean: g 5.365908382411402
	# real mean: r 68.24405045361863
	cfg.MODEL.PIXEL_MEAN = [87.693558721881, 5.365908382411402, 68.24405045361863] 
	#cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
	cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0 # loss function weighting, default is 1.0
	cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0 # loss function weighting, default is 1.0
	cfg.CUDNN_BENCHMARK = True # improves training speed 

	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set a custom testing threshold
	cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3     # non-max suppression IoU threshold
	
	return cfg