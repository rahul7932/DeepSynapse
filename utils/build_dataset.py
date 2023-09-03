import os, csv
from detectron2.structures import BoxMode

def build_dataset(dataDir, mode):
	# label directory that has csv for all tifs
	labelsDir = os.path.join(dataDir, "labels(xywh)")
	dataset_dicts = []
	# id for the image we are iterating on
	imgID = 1

	# image directory
	imgDir = os.path.join(dataDir, mode)
	print('Loading images from {} for {}'.format(imgDir, mode))

	# iterating through each image (filename) in sorted order
	for image in sorted(os.listdir(imgDir)):
		if image.endswith(".tif"):
			# a dictionary to store all of the characteristics of the image
			record = {}
			record['file_name'] = os.path.join(imgDir, image)
			record['image_id'] = imgID
			record['height'] = 1024
			record['width'] = 1024
			imgID += 1

			# first 8 chars of image file name (rrXXX_XX)
			imgSuffix = image[:-8]
			# same line print
			print(imgSuffix, end=', ')
			# getting the label file for each image
			labelFile = os.path.join(labelsDir, imgSuffix + ".csv")

			# finding the label file
			if os.path.exists(labelFile):	
				print('Label file exists')
				objs = []
				# opening the label file as data
				with open(labelFile, "r") as data:
					# reading the data as a DictReader where each entry in the DictReader is a dictionary with keys: x, y, w, h
					csvReader = csv.DictReader(data)
					# going through each row in the CSV
					for row in csvReader:
						# accessing each field from the CSV

						# WHAT DO WE DO WITH THESE VALS? ADD TO DICTIONARY?
						x = float(row['x'])
						y = float(row['y'])
						width = float(row['width'])
						height = float(row['height'])

						# prune spines that were oddly transformed
						# keep valid bboxes
						if width * height < 10000:
							ann = {'bbox': [x, y, width, height],'bbox_mode': BoxMode.XYWH_ABS,'category_id': 0}
							objs.append(ann)
				record['annotations'] = objs
				dataset_dicts.append(record)
			else:
				print('Does not have a label file')
	# a list of dictionaries where each dictionary holds keys: file name, image id, height, width, annotations
	return dataset_dicts