import numpy as np
import matplotlib.pyplot as plt



class KalmanFilter():
    def __init__(self):
        self.x = np.matrix('0. 0. 0. 0.').T
        self.P = np.matrix(np.eye(4)) * 1000  # initial uncertainty
        self.F = np.matrix('''
                          1. 0. 1. 0.;
                          0. 1. 0. 1.;
                          0. 0. 1. 0.;
                          0. 0. 0. 1.
                          ''')
        self.H = np.matrix('''
                          1. 0. 0. 0.;
                          0. 1. 0. 0.''')
        self.Q = np.matrix(np.eye(4))
        self.R = 0.01**2
        self.motion = np.matrix('0. 0. 0. 0.').T

    def kalman_xy(self, measurement_xy):
        """
        Parameters:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """
        self.x, self.P = self.kalman(self.x, self.P, measurement_xy, self.R, self.motion, self.Q, self.H, self.F)

    def kalman(self, x, P, measurement, R, motion, Q, H, F):
        '''
        Parameters:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

        Return: the updated and predicted new values for (x, P)

        See also http://en.wikipedia.org/wiki/Kalman_filter

        This version of kalman can be applied to many different situations by
        appropriately defining F and H
        '''
        # UPDATE x, P based on measurement m
        # distance between measured and current position-belief
        y = np.matrix(measurement).T - H * x
        S = H * P * H.T + R  # residual convariance
        K = P * H.T * S.I  # Kalman gain
        x = x + K * y
        I = np.matrix(np.eye(F.shape[0]))  # identity matrix
        P = (I - K * H) * P

        # PREDICT x, P based on motion
        x = F * x + motion
        P = F * P * F.T + Q

        return x, P


if __name__ == '__main__':

    kalman_filter = KalmanFilter()
    N = 20
    true_x = np.linspace(0.0, 10.0, N)
    true_y = true_x ** 2
    observed_x = true_x + 0.05 * np.random.random(N) * true_x
    observed_y = true_y + 0.05 * np.random.random(N) * true_y
    plt.plot(observed_x, observed_y, 'ro')
    result = []
    for meas in zip(observed_x, observed_y):
        kalman_filter.kalman_xy(np.array(meas).reshape(-1,2))
        result.append((kalman_filter.x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    plt.plot(kalman_x, kalman_y, 'g-')
    plt.show()