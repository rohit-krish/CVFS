"""
Created on Wed Jan  26 10:25:07 2022

@author: rohit krishna
@email : dev.rohitnp@gmail.com
"""
import numpy as np


def threshold_hysteresis(img: np.ndarray, lowThresholdRatio=0.05, highThresholdRatio=0.09, weak=np.int32(25)):
    '''
    Double threshold and Hysteresis
    '''

    # DOUBLE THRESHOLDING

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    # zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # HYSTERESIS

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (res[i, j] == weak):
                if (
                    (res[i+1, j-1] == strong) or (res[i+1, j] == strong) or
                    (res[i+1, j+1] == strong) or (res[i, j-1] == strong) or
                    (res[i, j+1] == strong) or (res[i-1, j-1] == strong) or
                    (res[i-1, j] == strong) or (res[i-1, j+1] == strong)
                ):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return res
