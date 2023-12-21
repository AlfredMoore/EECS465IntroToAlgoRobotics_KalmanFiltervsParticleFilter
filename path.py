import numpy as np
step = 0.1
# (0,0) -> (0,0.5)
map_phd1y = np.arange(0,0.5+step,step).reshape((1,-1))
map_phd1x = np.zeros_like(map_phd1y)
map_phd1theta = np.ones_like(map_phd1y) * np.pi/2
map_phd1 = np.concatenate((map_phd1x,map_phd1y,map_phd1theta),axis=0)
# print(map_phd1)

# (0,0.5) -> (8.5,0.5)
map_phd2x = np.arange(0,8.5+step,step).reshape((1,-1))
map_phd2y = np.ones_like(map_phd2x) * 0.5
map_phd2theta = np.ones_like(map_phd2x) * np.pi * 0
map_phd2 = np.concatenate((map_phd2x,map_phd2y,map_phd2theta),axis=0)
# print(map_phd2)

# (8.5,0.5) -> (8.5,11.0)
map_phd3y = np.arange(0.5,11.0+step,step).reshape((1,-1))
map_phd3x = np.ones_like(map_phd3y) * 8.5
map_phd3theta = np.ones_like(map_phd3y) * np.pi * 0.5
map_phd3 = np.concatenate((map_phd3x,map_phd3y,map_phd3theta),axis=0)
# print(map_phd3)

# (8.5,11.0) -> (16.0,11.0)
map_phd4x = np.arange(8.5,16.0+step,step).reshape((1,-1))
map_phd4y = np.ones_like(map_phd4x) * 11.0
map_phd4theta = np.ones_like(map_phd4x) * np.pi * 0
map_phd4 = np.concatenate((map_phd4x,map_phd4y,map_phd4theta),axis=0)

# (16.0,11.0) -> (16.0,0.5)
map_phd5y = np.arange(11.0,0.50-step,-step).reshape((1,-1))
map_phd5x = np.ones_like(map_phd5y) * 16.0
map_phd5theta = np.ones_like(map_phd5y) * np.pi * -0.5
map_phd5 = np.concatenate((map_phd5x,map_phd5y,map_phd5theta),axis=0)

# (16.0,0.5) -> (12.0,0.5)
map_phd6x = np.arange(16.0,12.0-step,-step).reshape((1,-1))
map_phd6y = np.ones_like(map_phd6x) * 0.5
map_phd6theta = np.ones_like(map_phd6x) * np.pi * -1
map_phd6 = np.concatenate((map_phd6x,map_phd6y,map_phd6theta),axis=0)

# (12.0,0.5) -> (12.0,4.0)
map_phd7y = np.arange(0.5,4.0+step,step).reshape((1,-1))
map_phd7x = np.ones_like(map_phd7y) * 12.0
map_phd7theta = np.ones_like(map_phd7y) * np.pi * 0.5
map_phd7 = np.concatenate((map_phd7x,map_phd7y,map_phd7theta),axis=0)

map_phd_path = np.concatenate((map_phd1,map_phd2,map_phd3,map_phd4,map_phd5,map_phd6,map_phd7), axis=1)
# print(map_phd.shape)
# pr2doorway_path: (2,1274)
# print(pr2doorway_path.shape)