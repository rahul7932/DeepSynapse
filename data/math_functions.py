import math

# calculate absolute distance between two 2D points (tuples)
def distance(t0, t1):
    return math.sqrt((t0[0] - t1[0])**2 + (t0[1] - t1[1])**2)

# calculate absolute distance between two 3D points (tuples)
def vectorLength(v0, v1):
    return math.sqrt((v0[0] - v1[0])**2 + (v0[1] - v1[1])**2 + (v0[2] - v1[2])**2)

# calculate midpoint between two 3D points
def midpoint(v0, v1):
    return ((v0[0] + v1[0]) * 1/2, (v0[1] + v1[1]) * 1/2, (v0[2] + v1[2]) * 1/2)


def boxCalc(base, tip, dist, width):
    angle = math.atan2(abs(tip[0]-base[0]),
                       abs(tip[1]-base[1])) * (180/math.pi)
    height = dist*1.5  # elongate
    shiftBox = width/2  # amount to shift box
    shiftOrigin = dist/4  # amount to shift origin
    config = 0

    # Config 1 and 2 are very similar
    # config 1: tip is to lower right of base
    if base[0] < tip[0] and base[1] < tip[1]:
        #print("Config 1")
        # origin = (tip[0], tip[1])　# for seeing original origin

        radian1 = (90-angle) * (math.pi/180)
        radian1 = (90-angle) * (math.pi/180)
        # translates origin for optimal bounding
        deltaX1 = shiftOrigin*math.cos(radian1)
        deltaY1 = shiftOrigin*math.sin(radian1)
        origin = (tip[0] + deltaX1, tip[1] + deltaY1)

        # translates box for optimal bounding
        radian2 = angle * (math.pi/180)
        deltaX2 = shiftBox*math.cos(radian2)
        deltaY2 = shiftBox*math.sin(radian2)
        origin = (origin[0]-deltaX2, origin[1]+deltaY2)

        angle = -1 * angle

        # corner calculations for bounding box annotations
        radian3 = (90-(-angle)) * (math.pi/180)
        deltaX3 = -1 * math.cos(radian3) * height
        deltaY3 = -1 * math.sin(radian3) * height

        radian4 = (90-(-angle)) * (math.pi/180)
        deltaX4 = math.sin(radian4) * width
        deltaY4 = -1 * math.cos(radian4) * width

        #corner = (origin[0]+deltaX3, origin[1]+deltaY3)

        corner = (origin[0]+deltaX3+deltaX4, origin[1]+deltaY3+deltaY4)
        config = 1

    # config 2: tip is to upper left of base
    elif base[0] > tip[0] and base[1] > tip[1]:
        #print("Config 2")
        radian1 = (90-angle) * (math.pi/180)
        # translates origin for optimal bounding
        deltaX1 = shiftOrigin*math.cos(radian1)
        deltaY1 = shiftOrigin*math.sin(radian1)
        origin = (base[0] + deltaX1, base[1] + deltaY1)

        # translates box for optimal bounding
        radian2 = angle * (math.pi/180)
        deltaX2 = shiftBox*math.cos(radian2)
        deltaY2 = shiftBox*math.sin(radian2)
        origin = (origin[0]-deltaX2, origin[1]+deltaY2)

        angle = -1 * angle

        # corner calculations for bounding box annotations
        radian3 = (90-(-angle)) * (math.pi/180)
        deltaX3 = -1 * math.cos(radian3) * height
        deltaY3 = -1 * math.sin(radian3) * height

        radian4 = (90-(-angle)) * (math.pi/180)
        deltaX4 = math.sin(radian4) * width
        deltaY4 = -1 * math.cos(radian4) * width

        #corner = (origin[0]+deltaX3, origin[1]+deltaY3)

        corner = (origin[0]+deltaX3+deltaX4, origin[1]+deltaY3+deltaY4)
        config = 2

    # Config 3 and 4 are very similar
    # config 3: tip is to upper right of base
    elif base[0] < tip[0] and base[1] > tip[1]:
        #print("Config 3")
        radian1 = angle * (math.pi/180)
        # translates origin for optimal bounding
        deltaX1 = shiftOrigin*math.sin(radian1)
        deltaY1 = shiftOrigin*math.cos(radian1)
        origin = (base[0] - deltaX1, base[1] + deltaY1)

        # translates box for optimal bounding
        radian2 = (90-angle) * (math.pi/180)
        deltaX2 = shiftBox*math.sin(radian2)
        deltaY2 = shiftBox*math.cos(radian2)
        origin = (origin[0]-deltaX2, origin[1]-deltaY2)

        angle = 1 * angle  # doesn't do anything just for mathematical reference

        # corner calculations for bounding box annotations
        radian3 = (angle) * (math.pi/180)
        deltaX3 = 1 * math.sin(radian3) * height
        deltaY3 = -1 * math.cos(radian3) * height

        radian4 = (angle) * (math.pi/180)
        deltaX4 = math.cos(radian4) * width
        deltaY4 = math.sin(radian4) * width

        #corner = (origin[0]+deltaX3, origin[1]+deltaY3)

        corner = (origin[0]+deltaX3+deltaX4, origin[1]+deltaY3+deltaY4)
        config = 3

    # config 4: tip is to lower left of base
    elif base[0] > tip[0] and base[1] < tip[1]:
        #print("Config 4")
        radian1 = angle * (math.pi/180)
        # translates origin for optimal bounding
        deltaX1 = shiftOrigin*math.sin(radian1)
        deltaY1 = shiftOrigin*math.cos(radian1)
        origin = (tip[0] - deltaX1, tip[1] + deltaY1)

        # translates box for optimal bounding
        radian2 = (90-angle) * (math.pi/180)
        deltaX2 = shiftBox*math.sin(radian2)
        deltaY2 = shiftBox*math.cos(radian2)
        origin = (origin[0]-deltaX2, origin[1]-deltaY2)

        angle = 1 * angle  # doesn't do anything just for mathematical reference

        # corner calculations for bounding box annotations
        radian3 = (angle) * (math.pi/180)
        deltaX3 = 1 * math.sin(radian3) * height
        deltaY3 = -1 * math.cos(radian3) * height

        radian4 = (angle) * (math.pi/180)
        deltaX4 = math.cos(radian4) * width
        deltaY4 = math.sin(radian4) * width

        #corner = (origin[0]+deltaX3, origin[1]+deltaY3)

        corner = (origin[0]+deltaX3+deltaX4, origin[1]+deltaY3+deltaY4)
        config = 4

    # Config 5 and 6 are very similar
    # config 5: vertical alignment
    elif base[0] == tip[0]:
        #print("Config 5")

        # tip directly above base
        if tip[1] < base[1]:

            origin = (base[0], base[1] + shiftOrigin)

            origin = (origin[0] - shiftBox, origin[1])

            angle = 0  # keep box vertical

        # tip directly below base
        else:

            origin = (tip[0], tip[1] + shiftOrigin)

            origin = (origin[0] - shiftBox, origin[1])

            angle = 0  # keep box vertical

        corner = (origin[0]+width, origin[1]-height)
        config = 5

    # config 6: horizontal alignment
    elif base[1] == tip[1]:
        #print("Config 6")
        # tip directly left of base
        if tip[0] < base[0]:

            origin = (tip[0] - shiftOrigin, tip[1])

            origin = (origin[0], origin[1] - shiftBox)

            angle = 90   # turn box on its side
        # tip directly right of base
        else:

            origin = (base[0] - shiftOrigin, base[1])

            origin = (origin[0], origin[1] - shiftBox)

            angle = 90   # turn box on its side

        corner = (origin[0]+height, origin[1]+width)
        config = 6

    height = -height  # flip box
    return origin, angle, height, corner, config


def calcCorners(origin, corner, angle, width, height):
    """
    Given the origin and corner (opposite to origin) coordinates, determine coordinates
    of the other two corners in order to determine min/max X/Y coords for no-angle box labels
    """
    height = abs(height)
    width = abs(width)
    # Case 1
    # Corner is to top-right of origin
    if corner[0] > origin[0] and corner[1] < origin[1]:

        if angle == 0:
            c1 = (origin[0] + width, origin[1])
            c2 = (origin[0], origin[1] - height)
        elif angle < 0:
            angle = abs(angle)
            rad1 = (90 - angle) * (math.pi/180)
            dX1 = -1 * math.cos(rad1) * height
            dY1 = -1 * math.sin(rad1) * height
            c1 = (origin[0] + dX1, origin[1] + dY1)

            rad2 = (angle) * (math.pi/180)
            dX2 = abs(math.cos(rad2)) * width
            dY2 = -1 * math.sin(rad2) * width
            c2 = (origin[0] + dX2, origin[1] + dY2)
        else:
            rad1 = (angle) * (math.pi/180)
            dX1 = abs(math.sin(rad1)) * height
            dY1 = -1 * math.cos(rad1) * height
            c1 = (origin[0] + dX1, origin[1] + dY1)

            rad2 = (90 - angle) * (math.pi/180)
            dX2 = abs(math.sin(rad2)) * width
            dY2 = abs(math.cos(rad2)) * width
            c2 = (origin[0] + dX2, origin[1] + dY2)

    # Case 2
    # Corner is to top-left of origin
    elif corner[0] < origin[0] and corner[1] < origin[1]:
        angle = abs(angle)

        rad1 = (90 - angle) * (math.pi/180)
        dX1 = -1 * math.cos(rad1) * height
        dY1 = -1 * math.sin(rad1) * height
        c1 = (origin[0] + dX1, origin[1] + dY1)

        rad2 = (angle) * (math.pi/180)
        dX2 = abs(math.cos(rad2)) * width
        dY2 = -1 * math.sin(rad2) * width
        c2 = (origin[0] + dX2, origin[1] + dY2)

    # Case 3
    # Corner is to bottom-right of origin
    elif corner[0] > origin[0] and corner[1] > origin[1]:
        angle = abs(angle)
        if angle == 0:
            c1 = (origin[0] + width, origin[1])
            c2 = (origin[0], origin[1] + height)
        elif angle == 90:
            c1 = (origin[0] + height, origin[1])
            c2 = (origin[0], origin[1] + width)
        else:
            rad1 = (angle) * (math.pi/180)
            dX1 = abs(math.sin(rad1)) * height
            dY1 = -1 * math.cos(rad1) * height
            c1 = (origin[0] + dX1, origin[1] + dY1)

            rad2 = (90 - angle) * (math.pi/180)
            dX2 = abs(math.sin(rad2)) * width
            dY2 = abs(math.cos(rad2)) * width
            c2 = (origin[0] + dX2, origin[1] + dY2)

    # elif corner[1] == origin[1]:

    return c1, c2


def fillROIs1(roisArr, base, tip, dist, width):
    """
    Takes in array of current rois in image, base, tip, dist, width regarding single sample.
    Returns roisArr incremented by 1 in the areas with pixels surrounded by the box created
    by sample measurements.
    """

    # now have origin and corner pixel of box, need to determine how to highlight only
    # pixels within the boundaries of the box
    origin, angle, height, corner, config = boxCalc(base, tip, dist, width)

    height = abs(height)  # just need height as a length measurement here
    # config 1
    if config == 1 or config == 2:
        angle = abs(angle)
        # for corner 2
        radian1 = (90-angle) * (math.pi/180)
        totalY1 = math.sin(radian1) * height
        totalX1 = math.cos(radian1) * height
        corner1 = (origin[0]-totalX1, origin[1]-totalY1)
        rate1 = totalY1//totalX1
        xmov1 = 1
        ymov1 = rate1

        radian2 = (angle) * (math.pi/180)
        totalY2 = math.sin(radian2) * width
        totalX2 = math.cos(radian2) * width
        corner2 = (origin[0]+totalX2, origin[1]-totalY2)
        rate2 = totalX2//totalY2  # note the switched axes
        xmov2 = rate2
        ymov2 = 1
        return (corner2)
