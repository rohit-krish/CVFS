import numpy as np
import cv2


def histEqualization(channel: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(channel.flatten(), 256, [0, 255])
    cdf = hist.cumsum()
    cdf_norm = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    channel_new = cdf_norm[channel.flatten()]
    channel_new = np.reshape(channel_new, channel.shape)
    return channel_new


img = cv2.imread('../../../assets/lung.jpg')
img = cv2.resize(img, (0, 0), fx=.3, fy=.3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

img[..., 0] = histEqualization(img[..., 0])

img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

cv2.imshow("res", img)
cv2.waitKey(0)
