import math


def distance(t0, t1):
    """Calculate absolute distance between two 2D points (tuples)."""
    return math.sqrt((t0[0] - t1[0])**2 + (t0[1] - t1[1])**2)


def vectorLength(v0, v1):
    """Calculate absolute distance between two 3D points (tuples)."""
    return math.sqrt((v0[0] - v1[0])**2 + (v0[1] - v1[1])**2 + (v0[2] - v1[2])**2)


def midpoint(v0, v1):
    """Calculate midpoint between two 3D points."""
    return ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2, (v0[2] + v1[2]) / 2)


def boxCalc(base, tip, dist, width):
    """Calculate bounding box configurations based on point relationships."""
    angle_rad = math.atan2(abs(tip[0] - base[0]), abs(tip[1] - base[1]))
    angle_deg = math.degrees(angle_rad)
    height = dist * 1.5
    shiftBox = width / 2
    shiftOrigin = dist / 4

    # Helper function for repetitive calculations
    def calculate_origin_corner(tip_or_base, shift_angle1, shift_angle2):
        radian1 = math.radians(90 - shift_angle1)
        deltaX1 = shiftOrigin * math.cos(radian1)
        deltaY1 = shiftOrigin * math.sin(radian1)
        temp_origin = (tip_or_base[0] + deltaX1, tip_or_base[1] + deltaY1)

        radian2 = math.radians(shift_angle2)
        deltaX2 = shiftBox * math.cos(radian2)
        deltaY2 = shiftBox * math.sin(radian2)

        final_origin = (temp_origin[0] - deltaX2, temp_origin[1] + deltaY2)

        radian3 = math.radians(90 + angle_deg)
        deltaX3 = -math.cos(radian3) * height
        deltaY3 = -math.sin(radian3) * height

        radian4 = math.radians(angle_deg)
        deltaX4 = math.sin(radian4) * width
        deltaY4 = -math.cos(radian4) * width

        corner = (final_origin[0] + deltaX3 + deltaX4,
                  final_origin[1] + deltaY3 + deltaY4)

        return final_origin, corner

    if base[0] < tip[0] and base[1] < tip[1]:  # Config 1
        origin, corner = calculate_origin_corner(tip, angle_deg, -angle_deg)
        config = 1

    elif base[0] > tip[0] and base[1] > tip[1]:  # Config 2
        origin, corner = calculate_origin_corner(base, angle_deg, -angle_deg)
        config = 2

    elif base[0] < tip[0] and base[1] > tip[1]:  # Config 3
        origin, corner = calculate_origin_corner(
            base, -angle_deg, angle_deg - 90)
        config = 3

    elif base[0] > tip[0] and base[1] < tip[1]:  # Config 4
        origin, corner = calculate_origin_corner(
            tip, -angle_deg, angle_deg - 90)
        config = 4

    elif base[0] == tip[0]:  # Config 5: Vertical alignment
        if tip[1] < base[1]:  # Tip is above
            origin = (base[0], base[1] + shiftOrigin)
        else:  # Tip is below
            origin = (tip[0], tip[1] + shiftOrigin)
        origin = (origin[0] - shiftBox, origin[1])
        corner = (origin[0] + width, origin[1] - height)
        config = 5

    elif base[1] == tip[1]:  # Config 6: Horizontal alignment
        if tip[0] < base[0]:  # Tip is left
            origin = (tip[0] - shiftOrigin, tip[1])
        else:  # Tip is right
            origin = (base[0] - shiftOrigin, base[1])
        origin = (origin[0], origin[1] - shiftBox)
        corner = (origin[0] + height, origin[1] + width)
        config = 6

    return origin, -angle_deg, -height, corner, config


def calcCorners(origin, corner, angle):
    """Calculate all four bounding box corners based on an origin, a corner and an angle."""
    width = distance((origin[0], origin[1]), (corner[0], origin[1]))
    height = distance((origin[0], origin[1]), (origin[0], corner[1]))
    angle_rad = math.radians(angle)

    # Helper function to calculate rotated points
    def rotate_point(cx, cy, px, py, a_rad):
        return (math.cos(a_rad) * (px-cx) - math.sin(a_rad) * (py-cy) + cx,
                math.sin(a_rad) * (px-cx) + math.cos(a_rad) * (py-cy) + cy)

    p1 = rotate_point(origin[0], origin[1],
                      origin[0] + width, origin[1], angle_rad)
    p2 = rotate_point(origin[0], origin[1], origin[0],
                      origin[1] - height, angle_rad)
    p3 = rotate_point(origin[0], origin[1], origin[0] +
                      width, origin[1] - height, angle_rad)

    return origin, p1, p2, p3
