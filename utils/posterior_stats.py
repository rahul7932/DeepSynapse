# posterior_stats.py

import csv
import json
import os
import sys

from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

currDir = os.path.dirname(os.path.realpath(__file__))
fields = ["Idx", "x", "y", "z", "cx", "cy", "cz"]


def containsTip(box, spineTip):
    if box[0] <= spineTip[0] <= box[2] and box[1] <= spineTip[1] <= box[3]:
        return True
    else:
        return False


def generatePredHistData():
    """
    Using test dump data along with spine labels, calculate a list where each entry specifies
    how many spines a particular bounding box encapsulates (expect values between 0 and 5)
    """
    imgToBox = {}
    with open("predBoxes_test.txt") as f:
        imgToBox = json.load(f)["data"]
    spinePerBox = []
    for imgName in imgToBox:
        boxes = imgToBox[imgName]
        expNum = imgName.split("_")[0]
        spineTips = []
        labelPath = os.path.join(currDir, "labels+", expNum, imgName+".csv")

        with open(labelPath) as data:
            csvReader = csv.DictReader(data)

            for row in csvReader:
                # uncleaned data

                x = int(row[fields[1]])
                y = int(row[fields[2]])
                spineTips.append((x, y))

        for box in boxes:
            assert box[0] < box[2] and box[1] < box[3], box
            numTips = 0
            for spineTip in spineTips:
                if containsTip(box, spineTip):
                    numTips += 1

            if numTips > 6:
                pass

            spinePerBox.append(numTips)
    return spinePerBox, imgToBox.keys()


def generateGTHistData(images):
    spinePerBox = []
    for imgName in images:
        dataPath = os.path.join(currDir, "mmdetection",
                                "training", "labels2D+_xyxy", imgName+".csv")
        boxes = list(csv.reader(open(dataPath)))[1:]
        expNum = imgName.split("_")[0]
        spineTips = []
        labelPath = os.path.join(currDir, "labels+", expNum, imgName+".csv")

        with open(labelPath) as data:
            csvReader = csv.DictReader(data)

            for row in csvReader:
                # uncleaned data
                x = int(row[fields[1]])
                y = int(row[fields[2]])
                spineTips.append((x, y))

        for boxPre in boxes:
            box = [
                int(boxPre[0]),
                int(boxPre[3]),
                int(boxPre[2]),
                int(boxPre[1]),
            ]

            assert box[0] <= box[2] and box[1] <= box[3], box
            numTips = 0
            for spineTip in spineTips:
                if containsTip(box, spineTip):
                    numTips += 1

            if numTips > 6:
                pass

            spinePerBox.append(numTips)

        if numTips > 6:
            pass

    return spinePerBox


def overlapBoxes():
    imgToBox = {}
    pathToPreds = os.path.join(currDir, 'output', 'fRCNN_X101',
                               "predBoxes_josh_fRCNN_X101_test_cropped.txt")
    with open(pathToPreds) as f:
        imgToBox = json.load(f)["data"]
    predBoxes = imgToBox[filename + '.tif']

    # ch2 naming shenanigans
    dataPath = os.path.join(currDir, "spineData", "imagesJosh2D_croppedResize",
                            "labels(xywh)", filename+".csv")
    GTBoxes = list(csv.reader(open(dataPath)))[1:]

    imagePath = os.path.join(currDir, "spineData/imagesJosh2D_croppedResize/test", filename+".tif")
    # imagePath = os.path.join(currDir,'spineData/images2D/test',filename+".tif")

    data = Image.open(imagePath)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(data)

    lw = 0.8  # linewidth

    for box in predBoxes:

        assert box[0] < box[2] and box[1] < box[3], box
        maxX = int(box[2])
        minX = int(box[0])
        maxY = int(box[3])
        minY = int(box[1])

        height = maxY - minY
        width = maxX - minX

        origin = (minX, minY)

        rect = patches.Rectangle(origin, width, height, linewidth=lw,
                                 facecolor='None', edgecolor=(1, 0, 1, 1), angle=0)
        ax.add_patch(rect)

    for boxPre in GTBoxes:
        box = [
            int(boxPre[0]),
            int(boxPre[1]),
            int(boxPre[2]),
            int(boxPre[3]),
        ]

        #assert box[0] <= box[2] and box[1] <= box[3], box
        minX = box[0]
        minY = box[1]
        width = box[2]
        height = box[3]

        origin = (minX, minY)

        # if height*width > 10000:
        # Create a Rectangle patch
        rect = patches.Rectangle(origin, width, height, linewidth=lw,
                                 edgecolor='None', facecolor=(1, 1, 0, 0.4), angle=0)
        ax.add_patch(rect)

    fuchsia_rects = patches.Patch(color='fuchsia', label='Pred')
    yellow_rects = patches.Patch(color='yellow', label='GT')

    ax.legend(handles=[fuchsia_rects, yellow_rects], loc='upper left')
    outputDir = os.path.join(os.path.dirname(
        currDir), 'results/detectron/fRCNN_X101/samples_test_josh_cropped')
    os.makedirs(outputDir, exist_ok=True)

    plt.savefig(os.path.join(outputDir, filename), format="tiff")


def generateHistogram(spinePerBoxPred, spinePerBoxGT):
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    n_bins = 50

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(spinePerBoxPred, bins=n_bins, range=(0, 6))
    axs[0].set_title("Pred SpinePerBox")
    axs[1].hist(spinePerBoxGT, bins=n_bins, range=(0, 6))
    axs[1].set_title("GT SpinePerBox")

    plt.savefig("spinePerBox_Hists.jpg")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        spinePerBoxPred, images = generatePredHistData()
        spinePerBoxGT = generateGTHistData(images)
        generateHistogram(spinePerBoxPred, spinePerBoxGT)
    else:
        filename = sys.argv[1]
        overlapBoxes()
