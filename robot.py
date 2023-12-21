'''
robot simulation model
'''
import numpy as np

class Robot:
    def __init__(self, kwargs) -> None:
        '''
        params: A, B, C, R, Q
        '''
        self.A = kwargs['A']
        self.B = kwargs['B']
        self.C = kwargs['C']
        
        self.R = kwargs['R']    # motion noise
        self.Q = kwargs['Q']    # sensor noise
        
    # def physical_model_no_noise(self, state, u) -> np.ndarray:
    #     '''
    #     physical model: x_t = A_t @ x_t-1 + B_t @ u_t + R_t
    #     state: [3 , M|1]
    #     u
    #     '''
    #     u = u.reshape((-1,1))
    #     new_state = self.A @ state + self.B @ u
    #     return new_state
        
        
    def physical_model_add_noise(self, state, u) -> np.ndarray:
        '''
        physical model: x_t = A_t @ x_t-1 + B_t @ u_t + R_t
        state: [3 , 1]
        u
        '''
        u = u.reshape((3,1))
        state = state.reshape((3,1))
        new_state = self.A @ state + self.B @ u
        sensor_noise = np.random.multivariate_normal(np.zeros(3), self.R)  # [3,]
        
        return new_state + sensor_noise.reshape((3,1))
    
    
    def add_sensor_noise(self, real_pose) -> np.ndarray:
        '''
        sensor gaussian noise
        
        real_pose: (3,1)
        '''
        gaussian_noise = np.random.multivariate_normal(np.zeros(3), self.Q)
        return real_pose + gaussian_noise.reshape((3,1))