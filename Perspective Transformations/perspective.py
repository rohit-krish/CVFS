# https://franklinta.com/2014/09/08/computing-css-matrix3d-transforms/
# https://github.com/OlehOnyshchak/ImageTransformations/blob/master/PerspectiveTransformation.ipynb

import numpy as np
import cv2

circles = np.zeros((4, 2), np.int32)
counter = 0
img = cv2.imread('../../assets/book-in-a-table.jpg')

img_copy = img.copy()

def getPerspectiveTransform(src, dst):
    print(src)
    print(dst)
    # Build the A matrix
    A = np.zeros((8, 8))
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]

    b = np.array(dst).reshape((8, 1))

    # Solve the linear system Ax = b

    # x = np.linalg.solve(A, b)
    '''or'''
    x = np.linalg.inv(A) @ b

    M = np.ones(9)
    M[:8] = x.flatten()
    return M.reshape((3, 3))


def onPressed(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        circles[counter] = x, y
        counter += 1


while 1:
    if cv2.waitKey(1) == ord('q'):
        break

    if counter == 4:
        width  = circles[1][0] - circles[0][0]
        height = circles[2][1] - circles[0][1]

        pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        # matrix = cv2.getPerspectiveTransform(pts1, pts2)
        matrix = getPerspectiveTransform(pts1, pts2)

        img = img_copy.copy()
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        cv2.imshow("Output", imgOutput)
        counter = 0

    for x in range(4):
        cv2.circle(
            img, (circles[x][0], circles[x][1]), 5, (0, 255, 0), cv2.FILLED
        )

    cv2.imshow("Original", img)
    cv2.setMouseCallback("Original", onPressed)
