import numpy as np

config_dic = {
    'A': np.eye(3),
    'B': np.eye(3),
    'C': np.eye(3),
    'R': 0.02 * np.eye(3),  # motion
    'Q': 0.02 * np.eye(3),  # sensor
    'sample_M': 200,
    'sample_mu': np.zeros(3),
    'sample_cov': 0.02 * np.eye(3),
}

color_dic = {
    'red': (1,0,0,1),
    'black': (0,0,0,1),
    'blue':(0,0,1,1)
}