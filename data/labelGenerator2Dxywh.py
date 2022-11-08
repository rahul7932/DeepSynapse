"""
Generate 2D bbox labels folder from 3D labels.
"""


import os
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage import io
from utils.transformations import boxCalc, distance
import numpy as np
import sys


WIDTH = 11

currDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
labelDirsOld = os.path.join(currDir, "labelsTransformed")
fields = ["Idx", "x", "y", "z", "cx", "cy", "cz"]
newFields = ["x1", "y1", "x2", "y2", "width", "height"]
# make directory for 2D labels
labelDirsNew = os.path.join(currDir, "labels2D")
try:
	os.mkdir(labelDirsNew)
except OSError as error:
	print("Already made this directory brother.") 


for oldLabelDir in os.listdir(labelDirsOld):


	# get sample data
	expDir = os.path.join(labelDirsOld, oldLabelDir)   #exp = "rr106b_s1" as reference

	for csvFile in os.listdir(expDir):
		if csvFile.endswith(".csv"):

			newFile = os.path.join(labelDirsNew, csvFile)
			newRows = []

			with open(os.path.join(expDir, csvFile), "r") as data:
				csvReader = csv.DictReader(data)

				for row in csvReader:
					index = int( row[fields[0]] )
					x = float( row[fields[1]] ) 
					y = float( row[fields[2]] )
					z = float( row[fields[3]] )
					cx = float( row[fields[4]] )
					cy = float( row[fields[5]] )
					cz = float( row[fields[6]] )
					layerbase = int(cz)
					layertip = int(z)
					base = (cx, cy)
					tip = (x, y)
					width = WIDTH
					dist = distance(base, tip)
					origin, angle, height, corner, config = boxCalc(base, tip, dist, width)

					# For 2D bounding box, DoFs are: origin, corner, width, height
					newRow = [origin[0], origin[1], corner[0], corner[1], width, abs(height)]
					newRows.append(newRow)

				
			with open(newFile, "w") as newCSVfile:
					csvwriter = csv.writer(newCSVfile)  
					csvwriter.writerow(newFields)
					csvwriter.writerows(newRows)

	"""
	# add all spines between layers rangeMin and rangeMax
	for i in range(2, len(fileData)):

		# Calculate measures for box
		sample = fileData[i-1] #choose csv file row index (since spine indices skip values, not reliable)
		sampleIdx = sample[0]
		x = int(sample[1])
		y = int(sample[2])
		z = int( sample[3])
		cx = int(sample[4])
		cy = int(sample[5])
		cz = int(sample[6])

		if ((z <= rangeMax and z >= rangeMin) and (cz <= rangeMax and z >= rangeMin)):
			layerbase = int(cz)
			layertip = int(z)
			base = (cx, cy)
			tip = (x, y)
			width = WIDTH
			dist = distance(base, tip)
			origin, angle, height, corner, config = boxCalc(base, tip, dist, width)



			# Create a Rectangle patch
			rect = patches.Rectangle(origin,width,height,linewidth=0.4,edgecolor='r',facecolor='none', angle=angle)
			ax.add_patch(rect)


			# Points for base/tip reference (debugging)
			point1 = patches.Circle(base, fill=False, radius=0.5)
			point2 = patches.Circle(tip, fill=False, radius=0.5)
			point3 = patches.Circle(origin, fill=False, radius=0.5)
			point4 = patches.Circle(corner, fill=False, radius=0.5)
			ax.add_patch(point1)
			ax.add_patch(point2)
			ax.add_patch(point3)
			ax.add_patch(point4)
			annotations.append((origin, corner))

			
			boundAdd = 50
			plt.text(origin[0]+width, origin[1]+width,sampleIdx, fontsize=6, color="yellow")

	"""









