"""
Created on Wed Jan  25 21:00:08 2022

@author: rohit krishna
@email : dev.rohitnp@gmail.com
"""
import numpy as np
from gaussian_blur import gaussianBlur
from to_gray import togray
from convolution import convolution


def sobelEdgeDetection(image: np.ndarray, sigma: int | float, image_format: str, filter_shape: int | None):
    img = togray(image, image_format)
    blurred = gaussianBlur(img, sigma, filter_shape=filter_shape)[1] / 255

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

    Ix = convolution(blurred, Kx)
    Iy = convolution(blurred, Ky)

    G = np.hypot(Ix, Iy)

    G = G / G.max() * 255

    theta = np.arctan2(Iy, Ix)

    return np.squeeze(G), np.squeeze(theta)
