import numpy as np
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix


def ComputeDesignMatrix(groundPoints, Xo, Yo, Zo, omega, phi, kappa, foc):
    """
        Compute the derivatives of the collinear law (design matrix)

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: The design matrix

        :rtype: np.array nx6

    """
    # initialization for readability
    # omega = self.exteriorOrientationParameters[3]
    # phi = self.exteriorOrientationParameters[4]
    # kappa = self.exteriorOrientationParameters[5]

    # Coordinates subtraction
    dX = groundPoints[:, 0] - Xo
    dY = groundPoints[:, 1] - Yo
    dZ = groundPoints[:, 2] - Zo
    dXYZ = np.vstack([dX, dY, dZ])

    rotationMatrixT = (Compute3DRotationMatrix(omega, phi, kappa)).T
    rotatedG = rotationMatrixT.dot(dXYZ)
    rT1g = rotatedG[0, :]
    rT2g = rotatedG[1, :]
    rT3g = rotatedG[2, :]

    focalBySqauredRT3g = foc / rT3g ** 2

    dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
    dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

    dgdX0 = np.array([-1, 0, 0], 'f')
    dgdY0 = np.array([0, -1, 0], 'f')
    dgdZ0 = np.array([0, 0, -1], 'f')

    # Derivatives with respect to X0
    dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
    dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

    # Derivatives with respect to Y0
    dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
    dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

    # Derivatives with respect to Z0
    dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
    dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

    dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
    dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
    dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

    gRT3g = dXYZ * rT3g

    # Derivatives with respect to Omega
    dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                      rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

    dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                      rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

    # Derivatives with respect to Phi
    dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                    rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

    dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                    rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

    # Derivatives with respect to Kappa
    dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                      rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

    dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                      rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

    # all derivatives of x and y
    dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdKappa]).T,
                   np.vstack([dydX0, dydY0, dydZ0, dydKappa]).T])

    a = np.zeros((2 * dd[0].shape[0], 4))
    a[0::2] = dd[0]
    a[1::2] = dd[1]

    return a
