import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_gray(img: np.ndarray, format: str):
    '''
    Algorithm:
    >>> 0.2989 * R + 0.5870 * G + 0.1140 * B 

    - Returns a gray image
    '''
    r_coef = 0.2989
    g_coef = 0.5870
    b_coef = 0.1140

    # r_coef = 0.2126
    # g_coef = 0.7152
    # b_coef = 0.0722

    if format.lower() == 'bgr':
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        return r_coef * r + g_coef * g + b_coef * b
    elif format.lower() == 'rgb':
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return r_coef * r + g_coef * g + b_coef * b
    else:
        raise Exception('Unsupported value in parameter \'format\'')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('../../assets/lenna.png')
    gray = to_gray(img, 'bgr')
    gray_flatten = gray.flatten()

    gray[gray < 150] = 0
    gray[gray >= 150] = 255

    print(gray_flatten.shape)

    plt.subplot(121)
    plt.title('image histogram')
    plt.hist(gray_flatten, bins='fd', rwidth=0.9)

    plt.subplot(122)
    plt.imshow(gray, cmap='gray')

    plt.show()
