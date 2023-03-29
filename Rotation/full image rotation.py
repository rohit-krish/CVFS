import numpy as np
import cv2
import math


def naive_image_rotate(image, degree):
    '''
    This function rotates the image around its center by amount of degrees
    provided. The rotated image will show the full image.
    '''
    rads = math.radians(degree)
    # In this case, we consider the rotated image to be different than input
    # image size which will depend on the rotation angle.

    height_rot_img = round(abs(image.shape[0]*math.cos(rads))) + round(abs(image.shape[1]*math.sin(rads)))
    width_rot_img = round(abs(image.shape[0]*math.sin(rads))) + round(abs(image.shape[1]*math.cos(rads)))

    rot_img = np.uint8(
        np.zeros((height_rot_img, width_rot_img, image.shape[2])))

    # Finding the center point of the original image
    cx, cy = (image.shape[1]//2, image.shape[0]//2)

    # Finding the center point of rotated image.
    midx, midy = (width_rot_img//2, height_rot_img//2)

    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            x = (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads)
            y = -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads)

            x = round(x)+cy
            y = round(y)+cx

            if (x >= 0 and y >= 0 and x < image.shape[0] and y < image.shape[1]):
                rot_img[i, j, :] = image[x, y, :]

    return rot_img


def main():
    image = cv2.imread('../../assets/lenna.png')
    rotated_image = naive_image_rotate(image, 45)
    # cv2.imshow("original image", image)
    cv2.imshow("rotated image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
