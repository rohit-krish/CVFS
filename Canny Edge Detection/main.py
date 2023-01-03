import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def togray(img: np.ndarray, format: str):
    '''
    Algorithm:
    >>> 0.2989 * R + 0.5870 * G + 0.1140 * B 

    - Returns a gray image
    '''
    if format.lower() == 'bgr':
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    elif format.lower() == 'rgb':
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        raise Exception('Unsupported value in parameter \'format\'')


def guassianBlur(img: np.ndarray, sigma: float | int, filter_shape: list | tuple | None = None):
    '''
    - Returns a list that contains the filter and resultant image

    * if filter_shape is None then it calculated automatically as below,

    >>> _ = 2 * int(4 * sigma + 0.5) + 1
    >>> filter_shape = [_, _]

    ### Example:
    >>> import matplotlib.pyplot as plt
    >>> from PIL import Image
    >>> img = np.array(Image.open('../../assets/lenna.png'))
    >>> g_filter, blur_image = GuassianBlur(img, 4)
    '''

    if filter_shape == None:
        _ = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [_, _]

    gaussian_filter = np.zeros((filter_shape[0], filter_shape[1]), np.float32)
    size_y = filter_shape[0] // 2
    size_x = filter_shape[1] // 2

    x, y = np.mgrid[-size_y:size_y+1, -size_x:size_x+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gaussian_filter = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    filtered = np.zeros(img.shape, dtype=np.float32)
    for c in range(3):
        filtered[:, :, c] = convolve(
            img[:, :, c], gaussian_filter)

    return gaussian_filter, filtered.astype(np.uint8)


def sobelEdgeDetection(image: np.ndarray, sigma: int | float, image_format: str, filter_shape: int | None):
    blurred = guassianBlur(image, sigma, filter_shape=filter_shape)[1] / 255
    image = togray(blurred, image_format)

    '''Gradient calculation / Sobel Filters'''
    Kx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], np.float32
    )

    Ky = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]], np.float32
    )

    Ix = convolve(image, Kx)
    Iy = convolve(image, Ky)

    G = np.hypot(Ix, Iy)

    G = G / G.max() * 255

    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, theta):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi    # max -> 180, min -> -180
    angle[angle < 0] += 180         # max -> 180, min -> 0

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = img[i, j-1]
                q = img[i, j+1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                r = img[i-1, j+1]
                q = img[i+1, j-1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                r = img[i-1, j]
                q = img[i+1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                r = img[i+1, j+1]
                q = img[i-1, j-1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    '''
    Double threshold and Hysteresis
    '''

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    # zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def cannyEdgeDetection(
    image: np.ndarray, sigma: int | float, filter_shape, image_format='rgb',
    lowthreshold: float | int = 0.05, highthreshold: float | int = 0.09
):
    G, theta = sobelEdgeDetection(image, sigma, image_format, filter_shape)
    img = non_max_suppression(G, theta)
    img, weak, strong = threshold(img, lowthreshold, highthreshold)
    return hysteresis(img, weak, strong)


if __name__ == '__main__':
    img = cv2.imread('../../assets/lenna.png')
    res = cannyEdgeDetection(img, 5, None, 'bgr')
    plt.imshow(res, cmap='gray')
    plt.show()
