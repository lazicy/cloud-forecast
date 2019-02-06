import cv2
import numpy as np
from lines import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from helpers import *


def separate_digits(frame):

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(grey, 180, 255, cv2.THRESH_BINARY)

    return img_bin


def prepare_for_roi(frame):

    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(frame, kernel, iterations=3)
    img = invert(dilated)
    for_roi = erode(img)

    return for_roi;

def select_roi(image_orig, image_bin, image_bin_clean):

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 90 and h > 18 and h < 50 and w > 1 and w < 50:

            region = image_bin_clean[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), [x + w, y+h]])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    br_coords = [region[1] for region in regions_array]

    return image_orig, sorted_regions, br_coords


def resize_region(region):

    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


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