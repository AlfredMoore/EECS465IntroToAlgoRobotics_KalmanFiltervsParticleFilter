from filters.KalmanFilter import KalmanFilter
from filters.ParticleFilter import ParticleFilter
from matplotlib.pyplot import plot
import pybullet as p
import time
import numpy as np
from pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM, PR2_URDF, DRAKE_PR2_URDF, \
    SIDE_HOLDING_LEFT_ARM, PR2_GROUPS, open_arm, get_disabled_collisions, REST_LEFT_ARM, rightarm_from_leftarm
from utils import *
from pybullet_tools.utils import *
from robot import Robot

from config import config_dic, color_dic
from path import map_phd_path
from plot import draw_traj

def main():
    map = 'map/map_phd.json'
    
    print(f'########## Start Running with Map {map} ##########')
    
    connect(use_gui=True)

    robots, obstacles = load_env(map)
    p.resetDebugVisualizerCamera( cameraDistance=10, cameraYaw=0, cameraPitch=271, cameraTargetPosition=[13,5,5])
    
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    
    target_path = map_phd_path                 #   [3, N]
    print("target path shape",target_path.shape)

    robot_sim = Robot(config_dic)

    # #################################################
    # wait_for_user()
    
    kf = KalmanFilter(config_dic)
    pf = ParticleFilter(config_dic)

    
    T = 0
    # motion noise
    target_pose = target_path[:,T].reshape((3,1))
    init_pose = robot_sim.physical_model_add_noise( target_pose, np.zeros((3,1)) )

    set_joint_positions(robots['pr2'], base_joints, init_pose)
    
    last_pose = init_pose                   # x_t-1
    
    print('##### Kalman Filter  : RED #####')
    # Kalman Filter ##################################################
    kf_real_pose = np.zeros_like(target_path)
    kf_sensor_pose = np.zeros_like(target_path)
    kf_mu_pose = np.zeros_like(target_path)

    kf_real_pose[:,T], kf_sensor_pose[:,T], kf_mu_pose[:,T] = init_pose.reshape(-1), init_pose.reshape(-1), init_pose.reshape(-1)

    t_kf = time.time()
    while T < target_path.shape[1]-1:
        # next state
        T += 1
        target_pose = target_path[:,T].reshape((3,1))
        u_t = target_pose - last_pose      # u_t
        real_pose = robot_sim.physical_model_add_noise( last_pose, u_t )     # x_t

        # self checking module
        # print(real_pose)
        # print(u_t)

        set_joint_positions(robots['pr2'], base_joints, real_pose)
        
        # get robot sensor pose
        real_pose = np.array(get_joint_positions(robots['pr2'], base_joints)).reshape((3,1))      # x_t
        sensor_pose = robot_sim.add_sensor_noise(real_pose)              # z_t
        
        kf_mu, kf.Sigma = kf.filter(mu=last_pose, Sigma=kf.Sigma, z=sensor_pose, u=u_t)
        last_pose = kf_mu

        kf_real_pose[:,T], kf_sensor_pose[:,T], kf_mu_pose[:,T] = real_pose.reshape(-1), sensor_pose.reshape(-1), kf_mu.reshape(-1)
        
        # print(f'\nT:{T}')
        # time.sleep(0.02)
    t_kf = time.time() - t_kf

    
    print('##### Particle Filter: BLUE #####')
    # wait_for_user()
    # Particle Filter ##################################################
    T = 0
    set_joint_positions(robots['pr2'], base_joints, init_pose)
    
    last_pose = init_pose                   # x_t-1

    pf_real_pose = np.zeros_like(target_path)
    pf_sensor_pose = np.zeros_like(target_path)
    pf_mu_pose = np.zeros_like(target_path)

    pf_real_pose[:,T], pf_sensor_pose[:,T], pf_mu_pose[:,T] = real_pose.reshape(-1), real_pose.reshape(-1), real_pose.reshape(-1)

    t_pf = time.time()
    while T < target_path.shape[1]-1:
        # next state
        T += 1
        target_pose = target_path[:,T].reshape((3,1))
        u_t = target_pose - last_pose      # u_t
        real_pose = robot_sim.physical_model_add_noise( last_pose, u_t )     # x_t

        # self checking module
        # print(real_pose)
        # print(u_t)

        set_joint_positions(robots['pr2'], base_joints, real_pose)
        
        # get robot sensor pose
        real_pose = np.array(get_joint_positions(robots['pr2'], base_joints)).reshape((3,1))      # x_t
        sensor_pose = robot_sim.add_sensor_noise(real_pose)              # z_t
        
        pf_mu = pf.filter(z=sensor_pose, u=u_t)
        last_pose = pf_mu

        pf_real_pose[:,T], pf_sensor_pose[:,T], pf_mu_pose[:,T] = real_pose.reshape(-1), sensor_pose.reshape(-1), pf_mu.reshape(-1)
        
        # wait_for_user()
        # print(f'\nT:{T}, \ntarget:{target_path[:,T]}, \nreal:{real_pose}, \nsensor:{sensor_pose}, \nKF:{pf_mu}')
        # print(kf_mu_pose)
        # print(kf_real_pose)
        # print(f'\nT:{T}')
        # time.sleep(0.02)
    t_pf = time.time() - t_pf


    print(f'computational time:\nkalman filter {t_kf}\nparticle filter {t_pf}')

    #  conclusion
    print("Kalman Filter traj......")
    for i in range(target_path.shape[1]):
        set_joint_positions(robots['pr2'], base_joints, kf_real_pose[:,i])
        time.sleep(0.05)

    print("Particle Filter traj......")
    for i in range(target_path.shape[1]):
        set_joint_positions(robots['pr2'], base_joints, pf_real_pose[:,i])
        time.sleep(0.05)


    draw_traj(target_path, 0.1, color_dic['black'])
    draw_traj(kf_mu_pose, 0.1, color_dic['red'])
    draw_traj(pf_mu_pose, 0.1, color_dic['blue'])


    wait_for_user()

    
        
        
if __name__ == '__main__':
    main()

        