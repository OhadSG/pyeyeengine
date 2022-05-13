import cv2
import scipy.ndimage
import numpy as np


# http://www.scipy-lectures.org/advanced/image_processing/#mathematical-morphology

def edges(img_binray, threshold=0): # replace without scipy if possible...
    grady, gradx = imradient(img_binray)
    return np.sqrt((grady ** 2 + gradx ** 2)) > threshold

def imradient(img): #gradient y, #gradient x
    return [scipy.ndimage.filters.sobel(np.float32(img), axis=0, mode="nearest"),
    scipy.ndimage.filters.sobel(np.float32(img), axis=1, mode="nearest")]


def dilate(img):
    return scipy.ndimage.binary_dilation(img)


def erode(img):
    return scipy.ndimage.binary_erosion(img)


def medianBlur(im_noise, kernel_size=3):
    return scipy.ndimage.median_filter(im_noise, kernel_size)


def findContours(binary_image):
    label_im, nb_labels = connected_components(edges(binary_image))
    cv2.imshow("label image", np.uint8(label_im)*255)
    cv2.waitKey(1)
    # create list of mx2 arrays [pts num, [X, Y]] for each contour
    return [np.expand_dims(np.fliplr(np.array(np.nonzero(label_im == label)).T), 1) for label in range(1, nb_labels+1)]


def connected_components(mask):
    label_im, nb_labels = scipy.ndimage.label(mask)
    return label_im, nb_labels
