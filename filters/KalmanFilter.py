import numpy as np


class KalmanFilter:
    def __init__(self, kwargs):
        """
        params: A,B,C,R,Q
        """
        # physical model
        self.A = kwargs['A']
        self.B = kwargs['B']
        self.C = kwargs['C']
        self.R = kwargs['R']    # motion noise
        self.Q = kwargs['Q']    # sensor noise
        self.Sigma = np.eye(3)

    def filter(self, **kwargs):
        """
        Kalman filter
        params: mu, Sigma, z, u
        return: next mu and next sigma
        """
        mu, Sigma, z, u = kwargs['mu'].reshape((3,1)), kwargs['Sigma'], kwargs['z'].reshape((3,1)), kwargs['u'].reshape((3,1))
        #prediction step
        mu_overline = self.A @ mu + self.B @ u
        Sigma_overline = self.A @ Sigma @ self.A.T + self.R
        #correction step
        K = Sigma_overline @ self.C.T @ np.linalg.inv( self.C @ Sigma_overline @ self.C.T + self.Q )

        mu_new = mu_overline + K @ (z - self.C @ mu_overline)
        Sigma_new = (np.eye(mu.shape[0]) - K @ self.C) @ Sigma_overline

        # print(mu_overline)
        # print(mu_new)
        # print(Sigma_overline)
        
        return mu_new, Sigma_new
