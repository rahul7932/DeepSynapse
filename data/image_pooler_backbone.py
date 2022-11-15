"""
Generate 2D max pooled images folder from 3D images.
Constrained so that only backbone and surrounding pixels are included.
Used for precisely constructing cropped data.

Preprocessing step.
TODO(vir): figure out if this is required, and where to place it
"""


import os
import csv
import SimpleITK as sitk
import numpy as np
from PIL import Image

# plot backbone points on image
currDir = os.path.dirname(os.path.realpath(__file__))
spineDataDir = os.path.join(currDir, 'spineData')
imgDirsOld = os.path.join(spineDataDir, "imagesOG(3D)")
bbDir = os.path.join(spineDataDir, "backbones")
imgDirsNew = os.path.join(spineDataDir, "images2D_test")

EXPAND = 40  # number of pixels to expand out from backbone, need to experiment

# create directories
try:
    os.mkdir(imgDirsNew)
    print('info:', imgDirsNew + " created")
except OSError as error:
    print(f"info: already made {imgDirsNew} directory brother")


def containedBackbone(minMaxesArr, i, j):
    contained = False
    for minMaxTup in minMaxesArr:
        bbMinX, bbMinY, bbMaxX, bbMaxY = minMaxTup
        if ((j >= bbMinX and j <= bbMaxX) and (i >= bbMinY and i <= bbMaxY)):
            contained = True
    return contained


for imgFile in os.listdir(imgDirsOld)[:1]:
    try:
        # load in backbone data
        imgName = imgFile.split("_")
        bbFile = "_".join(imgName[0:2] + ["bb.csv"])
        bbPath = os.path.join(bbDir, bbFile)
        bbData = []

        with open(bbPath) as backbone:
            reader = csv.reader(backbone)
            next(reader)
            for row in reader:
                edited = row

                edited[0] = int(edited[0])
                edited[1] = int(edited[1])
                edited[2] = int(edited[2])
                # edited[2] -= 1   # z is layer in image starting at 1, need to start at 0 for data.seek below
                bbData.append(edited)

        # now parse bbData to determine cropped area for each layer
        # layerDict:  layer number --> ((minX, minY), (maxX, maxY)) for that layer
        bbDict = {}
        maxX = 0
        maxY = 0
        minX = 1024
        minY = 1024

        for row in bbData:
            ID = row[3]
            if ID not in bbDict:
                bbDict[ID] = {}  # empty dict which is itself a alyer dict correpsonding to ID backbone

            layerDict = bbDict[ID]
            key = row[2]

            maxX = max(maxX, row[0])
            maxY = max(maxY, row[1])
            minX = min(minX, row[0])
            minY = min(minY, row[1])

            if key not in layerDict:

                layerDict[key] = ((row[0], row[1]), (row[0], row[1]))
            else:
                prevMinTuple, prevMaxTuple = layerDict[key]
                newMinTuple = (min(row[0], prevMinTuple[0]), min(row[1], prevMinTuple[1]))
                newMaxTuple = (max(row[0], prevMaxTuple[0]), max(row[1], prevMaxTuple[1]))

                layerDict[key] = (newMinTuple, newMaxTuple)

        imgPath = os.path.join(imgDirsOld, imgFile)
        data = Image.open(imgPath)
        h, w = np.shape(data)
        maxPool = np.zeros((h, w), dtype=np.uint16)

        img3D = sitk.ReadImage(imgPath)
        img_arr_3D = sitk.GetArrayFromImage(img3D)

        # every image is 1024x1024 width/height, number of layers will vary
        img_arr_3D = np.asarray(img_arr_3D)
        numLayers = np.shape(img_arr_3D)[0]

        for i in range(0, numLayers):

            # don't pool layers that don't contain the backbone
            relevantIDs = []
            contained = False
            for ID in bbDict:
                if i in bbDict[ID]:
                    contained = True
                    relevantIDs.append(ID)
            if not contained:
                continue

            ###
            data.seek(i)
            currLayer = np.array(data)
            width, height = np.shape(currLayer)
            # crop out non-backbone pixels in current layer

            minMaxes = []
            for ID in relevantIDs:
                bbMinTuple = bbDict[ID][i][0]
                bbMaxTuple = bbDict[ID][i][1]

                bbMinX = max(bbMinTuple[0]-EXPAND, 0)
                bbMinY = max(bbMinTuple[1]-EXPAND, 0)
                bbMaxX = min(bbMaxTuple[0]+EXPAND, 1024)
                bbMaxY = min(bbMaxTuple[1]+EXPAND, 1024)
                minMaxes.append((bbMinX, bbMinY, bbMaxX, bbMaxY))

            # handle edge cases, consider improving with numpy shorthand
            for i in range(width):
                for j in range(height):

                    # zero out pixels outside of backbone range
                    # ((j >= bbMinX and j <= bbMaxX) and (i >= bbMinY and i <= bbMaxY)):
                    if not containedBackbone(minMaxes, i, j):
                        currLayer[i][j] = 0

            maxPool = np.maximum(currLayer, maxPool)  # maxPool of layers seen so far and new layer

        # preview image to save
        # img2D = Image.fromarray(maxPool, mode="I;16")#.ravel())
        # img2DArr = np.asarray(img2D)
        # plt.imshow(img2DArr, cmap='gray')
        #
        # fig, ax = plt.subplots(1, figsize=(8, 8))
        # ax.imshow(img2DArr)
        #
        # for dataPoint in bbData:
        #     point = patches.Circle((dataPoint[0], dataPoint[1]), fill=False, radius=0.5)
        #     ax.add_patch(point)
        #
        # plt.show()

        imgPathNew = os.path.join(imgDirsNew, imgFile)
        # plt.imsave(imgPathNew, img2DArr, format='tiff')#, cmap='gray')

        I8 = (((maxPool - maxPool.min()) / (maxPool.max() - maxPool.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(I8)  # , mode='RGB')#, mode="I;16")
        img.save(imgPathNew)
        print(f'info: {imgFile} processed')

    except Exception as e:
        print(f'info: {imgFile} not processed: {e}')


print("info: done processing images")
