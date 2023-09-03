import os
import csv
from detectron2.structures import BoxMode


def build_dataset(dataDir, mode):
    """
    Construct a dataset for Detectron2.

    Parameters:
    - dataDir (str): The root directory containing the dataset.
    - mode (str): The type of dataset, e.g., "train", "val", "test".

    Returns:
    - List[Dict]: A list of dictionaries where each dictionary represents an image and its annotations.
    """

    # Directory containing label CSV files for each .tif image
    labelsDir = os.path.join(dataDir, "labels(xywh)")

    dataset_dicts = []  # List to store dictionaries for each image
    imgID = 1           # ID counter for the images

    # Directory containing the .tif images
    imgDir = os.path.join(dataDir, mode)
    print(f'Loading images from {imgDir} for {mode}')

    # Iterate over each image in the directory
    for image in sorted(os.listdir(imgDir)):
        if image.endswith(".tif"):
            record = {
                'file_name': os.path.join(imgDir, image),
                'image_id': imgID,
                'height': 1024,
                'width': 1024
            }
            imgID += 1

            # Extracting prefix of the image filename to match with label files
            imgPrefix = image[:-8]
            print(imgPrefix, end=', ')

            # Path to the label file for the current image
            labelFile = os.path.join(labelsDir, imgPrefix + ".csv")

            if os.path.exists(labelFile):
                print('Label file exists')
                objs = []

                with open(labelFile, "r") as data:
                    csvReader = csv.DictReader(data)

                    for row in csvReader:
                        x, y, width, height = float(row['x']), float(
                            row['y']), float(row['width']), float(row['height'])

                        # Filter out bounding boxes with areas larger than 10,000
                        if width * height < 10000:
                            ann = {
                                'bbox': [x, y, width, height],
                                'bbox_mode': BoxMode.XYWH_ABS,
                                'category_id': 0
                            }
                            objs.append(ann)
                record['annotations'] = objs
                dataset_dicts.append(record)
            else:
                print('Does not have a label file')

    return dataset_dicts
