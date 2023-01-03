from scipy.ndimage import convolve
from PIL import Image
import numpy as np


def GuassianBlur(img: np.ndarray, sigma: float | int, filter_shape: int | None = None):
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

    size_y = filter_shape[0] // 2
    size_x = filter_shape[1] // 2

    # gaussian_filter = np.zeros((filter_shape[0], filter_shape[1]), np.float32)
    # for x in range(-size, size+1):
    #     for y in range(-size, size+1):
    #         x1 = 2*np.pi*(sigma**2)
    #         x2 = np.exp(-(x**2 + y**2)/(2 * sigma**2))
    #         gaussian_filter[x+size, y+size] = (1/x1)*x2
    '''or'''

    x, y = np.mgrid[-size_y:size_y+1, -size_x:size_x+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gaussian_filter = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    filtered = np.zeros(img.shape, dtype=np.float32)
    for c in range(3):
        filtered[:, :, c] = convolve(
            img[:, :, c], gaussian_filter)

    return gaussian_filter, filtered.astype(np.uint8)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    img = np.array(Image.open('../../assets/lenna.png'))
    g_filter, blur_image = GuassianBlur(img, 4)
    plt.subplot(121)
    plt.imshow(g_filter, cmap='gray')
    plt.subplot(122)
    plt.imshow(blur_image)
    plt.show()
