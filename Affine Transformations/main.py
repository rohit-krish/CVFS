import numpy as np


def invertAffineTransform(M):
    # Extract the rotation and translation components of the affine matrix
    R = M[:2, :2]
    t = M[:2, 2]

    # Compute the inverse of the rotation matrix
    R_inv = np.linalg.inv(R)

    # Compute the inverse affine transformation matrix
    M_inv = np.zeros((2, 3), dtype=np.float32)
    M_inv[:2, :2] = R_inv
    M_inv[:2, 2] = -np.dot(R_inv, t)

    return M_inv


def warpAffine(image, M, output_shape):
    rows, cols, _ = image.shape
    out_rows, out_cols = output_shape[:2]

    output = np.zeros(output_shape, dtype=image.dtype)

    for out_row in range(out_rows):
        for out_col in range(out_cols):
            # Calculate the corresponding pixel coordinates in the input image
            in_col, in_row, _ = np.dot(M, [out_col, out_row, 1]).astype(int)

            # Check if the pixel coordinates are within the bounds of the input image
            if 0 <= in_row < rows and 0 <= in_col < cols:
                output[out_row, out_col] = image[in_row, in_col]

    return output


def getAffineTransform(src, dst):
    src = np.array([[x, y, 1] for (x, y) in src])

    # M = (np.linalg.inv(src.T @ src) @ src.T @ dst)

    M = np.linalg.inv(src) @ dst # because src is a square matrix
    ''' or '''
    # M = np.linalg.solve(src, dst)

    return M.T


if __name__ == '__main__':
    import cv2

    image = cv2.imread('../../assets/apple1.png')

    # a = np.radians(-10)
    # M = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 0]])
    M = np.float32([[1, 0, 100], [0, 1, 10], [0, 0, 0]])

    rows, cols = image.shape[:2]

    # out_rows, out_cols = rows + 100, cols + 100

    output = warpAffine(image, M, image.shape)
    # output = warpAffine(image, M, image.shape)

    M_inv = list(invertAffineTransform(M))
    M_inv.append([0, 0, 0])
    inv_transformed = warpAffine(output, M_inv, image.shape)

    cv2.imshow('Input', image)
    cv2.imshow('Output', output)
    cv2.imshow('Inverse', inv_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
