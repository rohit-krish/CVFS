import numpy as np


def adaptive_thresholdMean(img, block_size, c):
    # Check that the block size is odd and nonnegative
    assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

    # Calculate the local threshold for each pixel
    height, width = img.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Calculate the local threshold using a square neighborhood centered at (i, j)
            x_min = max(0, i - block_size // 2)
            y_min = max(0, j - block_size // 2)
            x_max = min(height - 1, i + block_size // 2)
            y_max = min(width - 1, j + block_size // 2)
            block = img[x_min:x_max+1, y_min:y_max+1]
            thresh = np.mean(block) - c
            if img[i, j] >= thresh:
                binary[i, j] = 255

    return binary

def adaptive_thresholdGaussian(img, block_size, c):
    # Check that the block size is odd and nonnegative
    assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"
    
    # Calculate the local threshold for each pixel using a Gaussian filter
    threshold = cv2.GaussianBlur(img, (block_size, block_size), 0)
    threshold = threshold - c
    
    # Apply the threshold to the input image
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[img >= threshold] = 255
    
    return binary


if __name__ == '__main__':
    import cv2

    # Load the input image as grayscale
    img = cv2.imread('../../assets/sample.png', cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding
    block_size = 3
    c = 1
    # binary = adaptive_thresholdMean(img, block_size, c)
    # bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


    binary = adaptive_thresholdGaussian(img, block_size, c)
    bin = cv2.adaptiveThreshold(img, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    # Display the binary image
    cv2.imshow('Binary', binary)
    cv2.imshow("Bin", bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
