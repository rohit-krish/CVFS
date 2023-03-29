import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import cv2

img = cv2.imread("../../../assets/lady.png", cv2.IMREAD_GRAYSCALE)

block_size = 16
n_blocks = (img.shape[0] // block_size, img.shape[1] // block_size)

img_eq = np.zeros_like(img)


for i in range(n_blocks[0]):
    for j in range(n_blocks[1]):
        block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        block_eq = exposure.equalize_hist(block, 256)
        img_eq[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block_eq * 255

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, 'gray')
axs[0].set_title('Original Image')

axs[1].imshow(img_eq, 'gray')
axs[1].set_title('Locally Equalized Image')

plt.tight_layout()
plt.show()
