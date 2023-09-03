"""
Generate 2D bbox labels folder from 3D labels.
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.transformations import boxCalc, distance

WIDTH = 11


def process_csv_data(exp_dir, csv_file):
    new_rows = []

    with open(os.path.join(exp_dir, csv_file), "r") as data:
        csv_reader = csv.DictReader(data)
        fields = ["Idx", "x", "y", "z", "cx", "cy", "cz"]

        for row in csv_reader:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            cx = float(row["cx"])
            cy = float(row["cy"])
            cz = float(row["cz"])

            base = (cx, cy)
            tip = (x, y)
            width = WIDTH
            dist = distance(base, tip)
            origin, _, height, corner, _ = boxCalc(base, tip, dist, width)

            new_row = [origin[0], origin[1], corner[0],
                       corner[1], width, abs(height)]
            new_rows.append(new_row)

    return new_rows


def main():
    curr_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    label_dirs_old = os.path.join(curr_dir, "labelsTransformed")
    label_dirs_new = os.path.join(curr_dir, "labels2D")
    new_fields = ["x1", "y1", "x2", "y2", "width", "height"]

    if not os.path.exists(label_dirs_new):
        os.mkdir(label_dirs_new)
        print("Directory created!")
    else:
        print("Directory already exists.")

    for old_label_dir in os.listdir(label_dirs_old):
        exp_dir = os.path.join(label_dirs_old, old_label_dir)

        for csv_file in os.listdir(exp_dir):
            if csv_file.endswith(".csv"):
                new_file = os.path.join(label_dirs_new, csv_file)
                new_rows = process_csv_data(exp_dir, csv_file)

                with open(new_file, "w") as new_csv_file:
                    csv_writer = csv.writer(new_csv_file)
                    csv_writer.writerow(new_fields)
                    csv_writer.writerows(new_rows)

    # The commented out portion (creating patches, annotations, etc.) was left out
    # because it's incomplete and seems to be for debugging purposes.


if __name__ == "__main__":
    main()
