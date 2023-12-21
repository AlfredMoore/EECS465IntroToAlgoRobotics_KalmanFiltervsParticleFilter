import numpy as np
from config import config_dic, color_dic
from utils import *


def draw_traj(path, radius=0.1, color=(0,0,0,1)):
    path[-1,:] = 0  # z = 0

    for i in range(path.shape[1]):
        draw_sphere_marker(tuple(path[:,i]), radius, color)


def main():
    pass


if __name__ == '__main__':
    pass