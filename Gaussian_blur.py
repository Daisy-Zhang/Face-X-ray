import const
import numpy

def MyGaussianBlur(M):
    G = const.GAUSSIAN_MATRIX

    row, col = M.shape

    M_ = numpy.zeros((row, col))

    for i in range(row):
        for j in range(col):
            if i == 0 or j == 0 or i == row - 1 or j == col - 1:
                M_[i][j] = M[i][j]
                continue
            ans = M[i - 1][j - 1] * G[0][0] + M[i - 1][j] * G[0][1] + M[i - 1][j + 1] * G[0][2] + M[i][j - 1] * G[1][0] + M[i][j] * G[1][1] + M[i][j + 1] * G[1][2] + M[i + 1][j - 1] * G[2][0] + M[i + 1][j] * G[2][1] + M[i + 1][j + 1] * G[2][2]
            M_[i][j] = ans

    return M_