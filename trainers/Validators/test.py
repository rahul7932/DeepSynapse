from Models.model_cfg import get_cfg_mod
from trainers.spineTrain import SpineTrainer
from detectron2.utils.logger import setup_logger
setup_logger()

import os, cv2
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from Models.model_cfg import *

def test(dataset_dicts, spine_metadata, cfg, outputDir, weightPath, vis=False, val=False):
	cfg = get_cfg_mod(val=val)
	cfg.TEST.DETECTIONS_PER_IMAGE = 500  # used to be 100, but many dense images have close to 300 spines
	cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, outputDir)
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'weights', weightPath)  # path to the model we just trained

	##### NMS configs
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set a custom testing threshold
	cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5     # non-max suppression IoU threshold

	predictor = DefaultPredictor(cfg)

	if vis: 
		for i, d in enumerate(random.sample(dataset_dicts, 1)): 
			print(d["file_name"])  
			d["file_name"] = "/media/ding/6a72c58a-f37b-40d5-b0a6-34ec4a223eae/home/ding/SpineQuant/detection/spineData/images2D_croppedBackbone/test/rr114a_s14_ch2.tif"
			im = cv2.imread(d["file_name"])
			outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
			v = Visualizer(im[:, :, ::-1], metadata=spine_metadata, scale=0.5)
			boxes = outputs["instances"].get_fields()["pred_boxes"]
			#print(np.shape(boxes))
			out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
			plt.imshow(out.get_image()[:, :, ::-1])
			plt.show()
			#plt.imsave(os.path.join(cfg.OUTPUT_DIR, "sample "+str(i+1)),out.get_image()[:, :, ::-1], format="tiff")


	from detectron2.evaluation import COCOEvaluator, inference_on_dataset
	from detectron2.data import build_detection_test_loader
	testMode = 'val' if val else 'test'
	evaluator = COCOEvaluator("spine_{}".format(testMode), cfg, False, output_dir=cfg.OUTPUT_DIR)
	val_loader = build_detection_test_loader(cfg, "spine_{}".format(testMode))
	eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)
	print(eval_results)