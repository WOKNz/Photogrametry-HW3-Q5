import numpy as np
import pandas as pd
from scipy.linalg import block_diag


def intersecXYZ(camerapoints, external_orientation):
    alist = (np.array([])).reshape(0, 3)
    llist = (np.array([])).reshape(0, 1)
    final_XYZ = (np.array([])).reshape(3, 0)
    for imgid in range(0, int(np.max(camerapoints[:, 1])), 1):
        for row in range(0, camerapoints.shape[0], 1):
            if int(camerapoints[row, 1]) == imgid + 1:
                v = camerapoints[row, 2:4]
                norm = (np.hstack((v * 0.001, -0.03))).reshape(3, 1)
                norm = norm / np.linalg.norm(norm)
                norm = norm.reshape(3, 1)
                a = np.identity(3) - np.dot(norm, norm.reshape(1, 3))
                o = external_orientation[int(camerapoints[row, 0]) - 1, 0:3]
                o = (o).reshape(3, 1)
                l1 = np.dot(a, o)
                alist = np.vstack((alist, a))
                llist = np.vstack((llist, l1))

        result_pnt = np.dot(np.linalg.inv(np.dot(alist.T, alist)), np.dot(alist.T, llist))
        final_XYZ = np.hstack((final_XYZ, result_pnt))
        alist = (np.array([])).reshape(0, 3)
        llist = (np.array([])).reshape(0, 1)
    return final_XYZ.T

# test part
# aprox_ext_rotation = np.loadtxt('data/approxEOP.txt')
# tie_points_pd = pd.read_csv('data/tiePointsSamples.txt', sep=' |\t', engine='python')
# tie_points = tie_points_pd.to_numpy()
#
# intersecXYZ(tie_points,aprox_ext_rotation)
