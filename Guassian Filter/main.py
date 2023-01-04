"""
Created on Wed Jan  2 10:11:08 2023

@author: rohit krishna
"""

from convolution import convolution
from PIL import Image
import numpy as np


def GuassianBlur(img: np.ndarray, sigma: float | int, filter_shape: list | tuple | None = None):
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

    # generating filter shape with the sigma(standard deviation) given, if filter_shape is None
    if filter_shape == None:
        _ = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [_, _]

    size_y = filter_shape[0] // 2
    size_x = filter_shape[1] // 2
    gaussian_filter = np.zeros((filter_shape[0], filter_shape[1]), np.float32)

    for x in range(-size_y, size_y+1):
        for y in range(-size_x, size_x+1):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2 + y**2)/(2 * sigma**2))
            gaussian_filter[x+size_x, y+size_y] = normal*exp_term

    '''or'''

    # x, y = np.mgrid[-size_y:size_y+1, -size_x:size_x+1]
    # normal = 1 / (2.0 * np.pi * sigma**2)
    # gaussian_filter = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    blurred = np.zeros(img.shape, dtype=np.float32)
    blurred = convolution(img, gaussian_filter)

    return gaussian_filter, blurred.astype(np.uint8)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open('../../assets/lenna.png'))
    g_filter, blur_image = GuassianBlur(img, 5)

    # plotting
    plt.subplot(121)
    plt.imshow(g_filter, cmap='gist_heat')  # viridis is also cool
    plt.subplot(122)
    plt.imshow(blur_image)
    plt.tight_layout()
    plt.show()
