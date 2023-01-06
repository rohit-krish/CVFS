"""
Created on Wed Jan  4 12:06:28 2023

@author: rohit krishna
@email : dev.rohitnp@gmail.com
"""

import numpy as np


def convolution(image: np.ndarray, kernel: list | tuple) -> np.ndarray:
    '''
    It is a "valid" Convolution algorithm implementaion.

    ### Example

    >>> import numpy as np
    >>> from PIL import Image
    >>>
    >>> kernel = np.array(
    >>>     [[-1, 0, 1],
    >>>     [-2, 0, 2],
    >>>     [-1, 0, 1]], np.float32
    >>> )
    >>> img = np.array(Image.open('./lenna.png'))
    >>> res = convolution(img, Kx)
    '''

    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape

    # if the image is gray then we won't be having an extra channel so handling it
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape

    y_strides = m_i - m_k + 1  # possible number of strides in y direction
    x_strides = n_i - n_k + 1  # possible number of strides in x direction

    img = image.copy()
    output_shape = (m_i-m_k+1, n_i-n_k+1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0  # taking count of the convolution operation being happening

    output_tmp = output.reshape(
        (output_shape[0]*output_shape[1], output_shape[2])
    )

    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i+m_k, j:j+n_k, c]

                output_tmp[count, c] = np.sum(sub_matrix * kernel)

            count += 1

    output = output_tmp.reshape(output_shape)

    return output
