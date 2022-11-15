import os
import csv
from detectron2.structures import BoxMode


def build_dataset(dataDir, mode):
    # label directory that has csv for all tifs
    labelsDir = os.path.join(dataDir, "labels(xywh)")
    dataset_dicts = []

    # image directory
    imgDir = os.path.join(dataDir, mode)
    print('info: loading images from {} for {}'.format(imgDir, mode))

    # iterating through each image (filename) in sorted order
    for index, image in enumerate(sorted(os.listdir(imgDir))):
        if image.endswith(".tif"):
            record = {}
            record['file_name'] = os.path.join(imgDir, image)
            record['image_id'] = index + 1
            record['height'] = 1024
            record['width'] = 1024

            # first 8 chars of image file name (rrXXX_XX)
            imgSuffix = image[:-8]
            print(f'info: processing image {imgSuffix}')

            # getting the label file for each image
            labelFile = os.path.join(labelsDir, imgSuffix + ".csv")

            # finding the label file
            if os.path.exists(labelFile):
                objs = []

                # opening the label file as data
                with open(labelFile, "r") as data:
                    for row in csv.DictReader(data):
                        x = float(row['x'])
                        y = float(row['y'])
                        width = float(row['width'])
                        height = float(row['height'])

                        # prune spines that were oddly transformed, keep valid bboxes
                        if width * height < 10000:
                            objs.append({'bbox': [x, y, width, height],
                                         'bbox_mode': BoxMode.XYWH_ABS,
                                         'category_id': 0})

                record['annotations'] = objs
                dataset_dicts.append(record)

            else:
                print(f'err: does not have a label file {labelFile}')

    # a list of dictionaries where each dictionary holds keys: file name, image id, height, width, annotations
    return dataset_dicts