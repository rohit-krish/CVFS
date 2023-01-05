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

    if filter_shape == None:
        # generating filter shape with the sigma(standard deviation) given
        _ = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [_, _]

    elif len(filter_shape) != 2:
        raise Exception('shape of argument `filter_shape` is not a supported')

    m, n = filter_shape

    m_half = m // 2
    n_half = n // 2

    gaussian_filter = np.zeros((m, n), np.float32)

    for y in range(-m_half, m_half):
        for x in range(-n_half, n_half):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term

    blurred = convolution(img, gaussian_filter)

    return gaussian_filter, blurred.astype(np.uint8)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open('../../assets/lenna.png'))
    g_filter, blur_image = GuassianBlur(img, 5, (40, 40))

    # plotting
    plt.subplot(121)
    plt.imshow(g_filter, cmap='gist_heat')  # viridis is also cool
    plt.subplot(122)
    plt.imshow(blur_image)
    plt.tight_layout()
    plt.show()
