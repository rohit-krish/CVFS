'''not complete'''
# https://shrishailsgajbhar.github.io/post/Image-Processing-Image-Rotation-Without-Library-Functions

import numpy as np
import cv2
import math


def naive_image_rotate_cropped(image, degree):
    '''
    This function rotates the image around its center by amount of degrees
    provided. The size of the rotated image is same as that of original image.
    '''
    # First we will convert the degrees into radians
    rads = math.radians(degree)

    # We consider the rotated image to be of the same size as the original
    rot_img = np.uint8(np.zeros(image.shape))

    # Finding the center point of rotated (or original) image.
    height = rot_img.shape[0]
    width = rot_img.shape[1]

    midx, midy = (width//2, height//2)

    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            x = (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads)
            y = -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads)

            x = round(x)+midx
            y = round(y)+midy

            if (x >= 0 and y >= 0 and x < image.shape[0] and y < image.shape[1]):
                rot_img[i, j, :] = image[x, y, :]

    return rot_img


def main():
    image = cv2.imread('../../assets/lenna.png')
    rotated_image = naive_image_rotate_cropped(image, 135)
    # cv2.imshow("original image", image)
    cv2.imshow("rotated image", rotated_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
