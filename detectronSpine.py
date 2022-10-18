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
from spine_utils import *
import argparse


def train(dataset_dicts, spine_metadata, cfg, outputDir, toResume=False, weightPath='model_0044999.pth', vis=False):
	if vis:
		################# GT visualization
		for d in dataset_dicts[:5]:
		    img = cv2.imread(d["file_name"], cv2.IMREAD_GRAYSCALE)
		    print("Visualizing GT data for {}".format(d["file_name"]))
		    print("Contains {} boxes".format(len(d["annotations"])))
		    visualizer = Visualizer(img[:, :], metadata=spine_metadata, scale=1)
		    out = visualizer.draw_dataset_dict(d)
		    plt.imshow(out.get_image()[:, :, ::-1])
		    plt.show()


	from spineTrain import SpineTrainer

	cfg = get_cfg_mod()
	
	cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, outputDir, 'weights')

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = SpineTrainer(cfg) 
	if not train:
		cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightPath)

	trainer.resume_or_load(resume=toResume)
	print('Resume? : ' + str(toResume))

	# manually set fan speed to high before training for temp control
	os.system("nvidia-settings -a '[gpu:0]/GPUFanControlState=1'")
	os.system("nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=80'")
	os.system("nvidia-settings -a '[fan:1]/GPUTargetFanSpeed=80'")

	trainer.train()

	# if training is manually stopped make sure to run the following command in terminal:
	# nvidia-settings -a '[gpu:0]/GPUFanControlState=0'
	os.system("nvidia-settings -a '[gpu:0]/GPUFanControlState=0'")






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
			imgName = d["file_name"].split("/")[-1]#.split(".")[0]
			#imgName = "_".join(imgName.split("_")[0:2])
			outputs = predictor(im)
			boxes = outputs["instances"].get_fields()["pred_boxes"]
			imgDict = {}
			imgToBox["data"][imgName] = []
			for box in boxes:
				imgToBox["data"][imgName].append(box.tolist())
		json.dump(imgToBox, testOut)
		#print(len(imgToBox.keys()))
		#print(len(imgToBox["data"].keys()))


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
	## test these out to pul out AP or some similar value for intermediate/final metric reporting to nni
	#print(eval_results.keys())
	#print(eval_results['bbox'].keys())
	#print(eval_results['bbox']['AP'])



if __name__ == '__main__':
	currDir = os.path.dirname(os.path.realpath(__file__))
	parser = argparse.ArgumentParser(description = 'Run training or inference with detectron2 on spine data')
	parser.add_argument('mode', metavar='mode', type=str,
                    help='mode (train / resume / val / test) \nval/test: runs inference and then evaluates detections (no output file)')
	parser.add_argument('dataDir', type=str, default='/spineData/images2D_croppedBackbone',
                    	help='Parent directory of dataset (starting from current "detection" directory)')
	parser.add_argument('outputDir', metavar='outDir', type=str,
                    help='output directory to write/read weights from, also where inference result will be output')

	parser.add_argument('--weightPath', type=str)
	parser.add_argument('--visualizeGT', action='store_true')
	parser.add_argument('--visualizeTestPreds', action='store_true')
	parser.add_argument('--generatePredFile', action='store_true')

	args = parser.parse_args()
	print(args)
	dataDir = os.path.join(currDir, args.dataDir)
	if args.mode in ['train', 'resume']:
		dataset, metadata, cfg = load_dataset(dataDir, 'train')
	else:
		dataset, metadata, cfg = load_dataset(dataDir, args.mode)

	print('Running {}'.format(args.mode))
	if args.mode in ['train', 'resume']:
		toResume = args.mode!='train'
		train(dataset, metadata, cfg, args.outputDir, toResume=toResume, vis=args.visualizeGT)
	elif args.mode in ['test', 'val']:
		assert args.weightPath is not None
		if args.generatePredFile:
			print('Generating pred file')
			temp = cfg.OUTPUT_DIR
			inference(dataset, cfg, args.outputDir, args.weightPath, val=args.mode=='val')
			cfg.OUTPUT_DIR = temp
		print('Generating metric data through COCOEvaluator')
		test(dataset, metadata, cfg, args.outputDir, args.weightPath, vis=args.visualizeTestPreds, val=args.mode=='val')