"""
Generate 2D max pooled images folder from 3D images. 
Constrained so that only backbone and surrounding pixels are included.
Used for precisely constructing cropped data.
"""

import os
import csv
import SimpleITK as sitk
import numpy as np
from PIL import Image

EXPAND = 40  # Number of pixels to expand out from backbone

def contained_backbone(min_maxes_arr, i, j):
    for min_max_tup in min_maxes_arr:
        bb_min_x, bb_min_y, bb_max_x, bb_max_y = min_max_tup
        if bb_min_x <= j <= bb_max_x and bb_min_y <= i <= bb_max_y:
            return True
    return False

def process_image(img_file, img_dirs_old, bb_dir, img_dirs_new):
    try:
        # Load backbone data
        img_name = img_file.split("_")
        bb_file = "_".join(img_name[0:2] + ["bb.csv"])
        bb_path = os.path.join(bb_dir, bb_file)
        bb_data = []
        with open(bb_path) as backbone:
            reader = csv.reader(backbone)
            next(reader)
            for row in reader:
                bb_data.append([int(item) for item in row[:3]])

        # Determine cropped area for each layer
        bb_dict = {}
        for row in bb_data:
            ID = row[3]
            layer_key = row[2]
            if ID not in bb_dict:
                bb_dict[ID] = {}
            min_max_coords = bb_dict[ID].get(layer_key, ((1024, 1024), (0, 0)))
            bb_dict[ID][layer_key] = (
                (min(row[0], min_max_coords[0][0]), min(row[1], min_max_coords[0][1])),
                (max(row[0], min_max_coords[1][0]), max(row[1], min_max_coords[1][1]))
            )

        img_path = os.path.join(img_dirs_old, img_file)
        img_3D = sitk.ReadImage(img_path)
        img_arr_3D = sitk.GetArrayFromImage(img_3D)
        num_layers = img_arr_3D.shape[0]
        max_pool = np.zeros(img_arr_3D.shape[1:], dtype=np.uint16)  

        for layer in range(num_layers):
            relevant_ids = [ID for ID in bb_dict if layer in bb_dict[ID]]
            if not relevant_ids:
                continue

            curr_layer = img_arr_3D[layer]
            min_maxes = []
            for ID in relevant_ids:
                bb_min_coords, bb_max_coords = bb_dict[ID][layer]
                bb_min_x = max(bb_min_coords[0] - EXPAND, 0)
                bb_min_y = max(bb_min_coords[1] - EXPAND, 0)
                bb_max_x = min(bb_max_coords[0] + EXPAND, 1024)
                bb_max_y = min(bb_max_coords[1] + EXPAND, 1024)
                min_maxes.append((bb_min_x, bb_min_y, bb_max_x, bb_max_y))

            for i in range(curr_layer.shape[0]):
                for j in range(curr_layer.shape[1]):
                    if not contained_backbone(min_maxes, i, j):
                        curr_layer[i][j] = 0
            max_pool = np.maximum(curr_layer, max_pool)

        I8 = (((max_pool - max_pool.min()) / (max_pool.max() - max_pool.min())) * 255.9).astype(np.uint8)
        img_2D = Image.fromarray(I8)
        img_2D.save(os.path.join(img_dirs_new, img_file))
        print(f"{img_file} processed")
    except Exception as e:
        print(f"{img_file} not processed due to error: {e}")

def main():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    spine_data_dir = os.path.join(curr_dir, 'spineData')
    img_dirs_old = os.path.join(spine_data_dir, "imagesOG(3D)")
    bb_dir = os.path.join(spine_data_dir, "backbones")
    img_dirs_new = os.path.join(spine_data_dir, "images2D_test")

    if not os.path.exists(img_dirs_new):
        os.mkdir(img_dirs_new)
        print(f"{img_dirs_new} created!")

    for img_file in os.listdir(img_dirs_old)[:1]:
        process_image(img_file, img_dirs_old, bb_dir, img_dirs_new)

    print("Done processing")

if __name__ == "__main__":
    main()



