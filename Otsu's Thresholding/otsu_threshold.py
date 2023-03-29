# https://en.wikipedia.org/wiki/Otsu%27s_method

import numpy as np


def _compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1


def otsuThresholding(img):
    threshold_range = range(np.max(img)+1)
    criterias = np.array([_compute_otsu_criteria(img, th) for th in threshold_range])

    # best threshold is the one minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]

    ret, binary = cv2.threshold(img, best_threshold, 255, cv2.THRESH_BINARY)

    return binary

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread('../../assets/lenna.png', cv2.IMREAD_GRAYSCALE)

    binary = otsuThresholding(img)

    cv2.imshow("img", img)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
