import cv2
import numpy as np
import colorsys


from decimal import Decimal

all_blue_rps = []
all_blue_lps = []

all_green_rps = []
all_green_lps = []



def separate_lines(frame, color):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    up_limit, down_limit = get_hsv_value(color)
    # print(up_limit, down_limit)

    mask = cv2.inRange(hsv_frame, down_limit, up_limit)
    separated = cv2.bitwise_and(frame, frame, mask=mask)

    return separated


def get_hsv_value(color):

    if color == 'blue':
        rgb = np.array([0, 0, 1])
    elif color == 'green':
        rgb = np.array([0, 1, 0])
    elif color == 'white':
        rgb = np.array([1, 1, 1])
    else:
        return 0, 0

    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    hsv_up = np.array([179 * hsv[0] + 10, 255 * hsv[1], 255 * hsv[2]])
    hsv_down = np.array([179 * hsv[0] - 10, 100 * hsv[1], 100 * hsv[2]])

    # print(hsv_up)
    # print(hsv_down)

    return hsv_up, hsv_down


# returns coordinates of the lines
def hough(processed, frame):

    edges = cv2.Canny(processed, 75, 150, apertureSize=3)
    # cv2.imshow('canny', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=350)
    for x1, y1, x2, y2 in lines[0]:
       cv2.line(processed, (x1, y1), (x2, y2), (0, 0, 255), 3)


    return np.array([x1, y1]), np.array([x2, y2])


def calculate_definite_lines(color):

    if color == 'blue':
        left_p = all_blue_lps
        right_p = all_blue_rps
    elif color == 'green':
        left_p = all_green_lps
        right_p = all_green_rps

    x_min = left_p[0][0]
    y_max = left_p[0][1]
    # find min for x and max for y
    for x, y in left_p:

        if x < x_min:
            x_min = x

        if y > y_max:
            y_max = y

    z_max = right_p[0][1]
    v_min = right_p[0][1]
    # find min for x and max for y
    for z, v in right_p:

        if z > z_max:
            z_max = z

        if v < v_min:
            v_min = v

    return np.array([x_min-5, y_max]), np.array([z_max+5, v_min])


def lin_equ(l1, l2):
    #line encoded as l=(x,y)
    k = ((l2[1]-l1[1]))/(l2[0]-l1[0])
    n =(l2[1]-(k*l2[0]))
    return k, n

# this method is ruled out as untrustable :(
def check_if_on_line(p, k, n, min_x, max_x):

    dif = abs(p[1] - (k*p[0] + n))
    print('dif: ', dif)

    if dif < 1.2  and p[0] >= min_x and p[0] <= max_x:
        print('crossed')
        return True
    else:

        return False

# distanca p3 od lajne koja spaja p1 i p2
def check_distance_from_line(p3, p1, p2):

    d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    # print("distens: ", d)
    return d;




def work_with_lines(frame):
    blue = separate_lines(frame, 'blue')
    green = separate_lines(frame, 'green')

    grey_b = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    grey_g = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

    ret, bin_b = cv2.threshold(grey_b, 20, 255, cv2.THRESH_BINARY)
    ret, bin_g = cv2.threshold(grey_g, 20, 255, cv2.THRESH_BINARY)

    blur_b = cv2.medianBlur(bin_b, 5)
    blur_g = cv2.medianBlur(bin_g, 5)

    blueP1, blueP2 = hough(blur_b, frame)
    all_blue_lps.append(blueP1)
    all_blue_rps.append(blueP2)

    greenP1, greenP2 = hough(blur_g, frame)
    all_green_lps.append(greenP1)
    all_green_rps.append(greenP2)


def invert(image):
    return 255-image

