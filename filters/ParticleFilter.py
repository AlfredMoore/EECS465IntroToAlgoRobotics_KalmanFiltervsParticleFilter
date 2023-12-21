import numpy as np


class ParticleFilter:
    def __init__(self, kwargs):
        '''
        physical model: x_t = A_t @ x_t-1 + B_t @ u_t + R_t
        sensor model: z_t = C_t @ x_t + Q_t
        
        params: A, B, C, R, Q, sample_M, sample_mu, sample_cov
        '''
        
        self.sample_M = kwargs['sample_M']  # particle number
        self.A = kwargs['A']
        self.B = kwargs['B']
        self.C = kwargs['C']
        self.R = kwargs['R']    # motion noise
        self.Q = kwargs['Q']    # sensor noise
        
        # initialize particles according to map size
        self.sample_mu = kwargs['sample_mu']      # [3,]
        self.sample_cov = kwargs['sample_cov']    # [3,3]
        
        # we just need 2D position like (x,y)
        map_center = (12,5.5)
        map_extend = (12,5.5)
        init_particle_x = np.random.rand(1,self.sample_M)
        init_particle_y = np.random.rand(1,self.sample_M)
        init_particle_theta = np.zeros_like(init_particle_x)
        init_particle_mat = np.concatenate((init_particle_x,init_particle_y,init_particle_theta),axis=0)
        print("sample number:",init_particle_mat.shape)

        self.particle_mat = init_particle_mat   # should be [3, M]
        sample_noise = np.random.multivariate_normal(self.sample_mu, self.sample_cov, self.sample_M)   # [M, 2]
        sample_noise = np.transpose(sample_noise)     # [2, M]
        self.particle_mat = self.particle_mat + sample_noise     # particle_mat: [2, sample_M]


    def filter(self, **kwargs) -> np.ndarray:
        """
        Particle filter
        physical model: x_t = A_t @ x_t-1 + B_t @ u_t + R_t
        sensor model: z_t = C_t @ x_t + Q_t
        
        params: z, u
        return: next mu and next sigma
        """
        z, u = kwargs['z'].reshape((3,1)), kwargs['u'].reshape((3,1))
        
        # sample
        self.particle_mat = self.A @ self.particle_mat + self.B @ u     # particle_mat: [3, sample_M]
        
        # TODO: sample noise or not?
        # sample_noise = np.random.multivariate_normal(self.sample_mu, self.sample_cov, self.sample_M)   # [M, 3]
        # sample_noise = np.transpose(sample_noise)     # [3, M]
        # self.particle_mat = self.particle_mat + sample_noise   # particle_mat: [3, sample_M]

        # weight based on a Heuristic Function
        lowerbould = 10/self.sample_M
        errors = np.sum( np.square(self.particle_mat - z), axis=0)    # [1, M]
        errors = np.maximum(errors, lowerbould)     # in case of inf
        self.weights_mat = 1 / errors    # [1, M]


        # resample: Systematic resampling
        indexes = self.systematic_resample(self.weights_mat)
        self.particle_mat = self.particle_mat[:, indexes]
        
        mu_new = np.mean(self.particle_mat, axis=1)
        
        return mu_new.reshape((3,1))   # [3,]

            
    def systematic_resample(self, weights_mat) -> np.ndarray:
        '''
        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.
        
        return indexes
        '''
        N = self.sample_M

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.rand() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights_mat)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
