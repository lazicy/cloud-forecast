import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255-image


def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def erode_small(image):
    kernel = np.ones((1,1)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)



def dilate_small(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def blur_avg(image):

    return cv2.blur(image, (2, 2))
