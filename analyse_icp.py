import numpy as np
from utils.plot_util import *

poses_icp = np.load('data/20' + '/poses/initialised_poses_icpo.npy')
X_icp = np.load('data/20' + '/poses/X_icp.npy')
Y_icp = np.load('data/20' + '/poses/Y_icp.npy')
thetas_icp = np.load('data/20' + '/poses/theta_icp.npy')

poses_mm = np.load('data/20' + '/poses/initialised_poses_motion_model.npy')
X_mm = np.load('data/20' + '/poses/X_motion_model.npy')
Y_mm = np.load('data/20' + '/poses/Y_motion_model.npy')
thetas_mm = np.load('data/20' + '/poses/theta_motion_model.npy')

plot_comparisons(X_icp[2800:3000], Y_icp[2800:3000], 'mm', X_mm[2800:3000], Y_mm[2800:3000], 'icp', 'testing', 'comparison.png')
