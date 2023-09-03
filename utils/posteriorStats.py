import os
import json
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Global Variables
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
FIELDS = ["Idx", "x", "y", "z", "cx", "cy", "cz"]


def contains_tip(box, spine_tip):
    """Check if the bounding box contains the spine tip."""
    return box[0] <= spine_tip[0] <= box[2] and box[1] <= spine_tip[1] <= box[3]


def read_csv_labels(img_name, exp_num):
    """Utility function to read CSV spine labels."""
    label_path = os.path.join(CURR_DIR, "labels+", exp_num, f"{img_name}.csv")
    spine_tips = []
    with open(label_path) as data:
        csv_reader = csv.DictReader(data)
        for row in csv_reader:
            x, y = int(row[FIELDS[1]]), int(row[FIELDS[2]])
            spine_tips.append((x, y))
    return spine_tips


def generate_pred_hist_data():
    """Calculate the number of spines each bounding box in prediction data encapsulates."""
    with open("predBoxes_test.txt") as f:
        img_to_box = json.load(f)["data"]

    spine_per_box = []
    for img_name, boxes in img_to_box.items():
        exp_num = img_name.split("_")[0]
        spine_tips = read_csv_labels(img_name, exp_num)

        for box in boxes:
            num_tips = sum(contains_tip(box, tip) for tip in spine_tips)
            spine_per_box.append(num_tips)

    return spine_per_box, list(img_to_box.keys())


def generate_gt_hist_data(images):
    """Calculate the number of spines each bounding box in ground truth data encapsulates."""
    spine_per_box = []
    for img_name in images:
        data_path = os.path.join(
            CURR_DIR, "mmdetection", "training", "labels2D+_xyxy", f"{img_name}.csv")
        boxes = list(csv.reader(open(data_path)))[1:]
        exp_num = img_name.split("_")[0]
        spine_tips = read_csv_labels(img_name, exp_num)

        for box_pre in boxes:
            box = [int(coord) for coord in box_pre]
            num_tips = sum(contains_tip(box, tip) for tip in spine_tips)
            spine_per_box.append(num_tips)

    return spine_per_box


def overlap_boxes(filename):
    """Display overlapping bounding boxes from predictions and ground truth on an image."""
    path_to_preds = os.path.join(
        CURR_DIR, 'output', 'fRCNN_X101', "predBoxes_josh_fRCNN_X101_test_cropped.txt")
    with open(path_to_preds) as f:
        img_to_box = json.load(f)["data"]

    pred_boxes = img_to_box[f"{filename}.tif"]
    data_path = os.path.join(
        CURR_DIR, "spineData", "imagesJosh2D_croppedResize", "labels(xywh)", f"{filename}.csv")
    gt_boxes = list(csv.reader(open(data_path)))[1:]
    image_path = os.path.join(
        CURR_DIR, "spineData/imagesJosh2D_croppedResize/test", f"{filename}.tif")
    visualize_overlap(image_path, pred_boxes, gt_boxes)


def visualize_overlap(image_path, pred_boxes, gt_boxes):
    """Utility function to visualize overlaps."""
    data = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(data)

    for box in pred_boxes:
        draw_rectangle(ax, box, edge_color=(1, 0, 1, 1))

    for box in gt_boxes:
        draw_rectangle(ax, box, face_color=(1, 1, 0, 0.4))

    fuchsia_rects = patches.Patch(color='fuchsia', label='Pred')
    yellow_rects = patches.Patch(color='yellow', label='GT')
    ax.legend(handles=[fuchsia_rects, yellow_rects], loc='upper left')
    output_dir = os.path.join(os.path.dirname(
        CURR_DIR), 'results/detectron/fRCNN_X101/samples_test_josh_cropped')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(
        output_dir, image_path.split('/')[-1]), format="tiff")
    plt.show()


def draw_rectangle(ax, box, edge_color=None, face_color='None', lw=0.8):
    """Utility function to draw a rectangle on an axis."""
    origin = (box[0], box[1])
    width, height = box[2] - box[0], box[3] - box[1]
    rect = patches.Rectangle(
        origin, width, height, linewidth=lw, edgecolor=edge_color, facecolor=face_color)
    ax.add_patch(rect)


def generate_histogram(spine_per_box_pred, spine_per_box_gt):
    """Generate histogram comparing number of spines per bounding box for predictions and ground truth."""
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    n_bins = 50
    axs[0].hist(spine_per_box_pred, bins=n_bins, range=(0, 6))
    axs[0].set_title("Pred SpinePerBox")
    axs[1].hist(spine_per_box_gt, bins=n_bins, range=(0, 6))
    axs[1].set_title("GT SpinePerBox")
    plt.savefig("spinePerBox_Hists.jpg")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        spine_per_box_pred, images = generate_pred_hist_data()
        spine_per_box_gt = generate_gt_hist_data(images)
        generate_histogram(spine_per_box_pred, spine_per_box_gt)
    else:
        filename = sys.argv[1]
        overlap_boxes(filename)
