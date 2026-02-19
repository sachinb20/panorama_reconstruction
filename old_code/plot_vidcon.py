from transforms3d.euler import mat2euler
from transforms3d.euler import euler2mat
from math import pi
import numpy as np
from load_data import read_data

dataset="3"
cfile = "../data/trainset/cam/cam" + dataset + ".p"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

viconrots= read_data(vfile)
print(viconrots.keys())
print(viconrots['rots'][0])

vicon_euler = np.array([mat2euler(viconrots['rots'][:, :, i], 'syxz') for i in range(viconrots['rots'].shape[2])])
print(vicon_euler.shape)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(vicon_euler[:, 0])
axs[0].set_ylabel('Roll (rad)')
axs[0].set_title('Vicon Euler Angles')

axs[1].plot(vicon_euler[:, 1])
axs[1].set_ylabel('Pitch (rad)')

axs[2].plot(vicon_euler[:, 2])
axs[2].set_ylabel('Yaw (rad)')
axs[2].set_xlabel('Sample Index')

plt.tight_layout()
plt.show()