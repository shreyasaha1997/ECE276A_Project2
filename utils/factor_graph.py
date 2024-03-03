import gtsam
from gtsam import Pose2, GaussNewtonParams, BetweenFactorPose2
from .plot_util import *
from .lidar_utils import *
import numpy as np
from tqdm import tqdm
import os


def Vector3(x, y, z): return np.array([x, y, z])

def get_x_y_theta(pose):
    x = pose[0][3]
    y = pose[1][3]
    Rt = pose[:3,:3]
    theta = np.arctan2(Rt[1, 0], Rt[0, 0])
    return x, y, theta

def create_factor_graph(args, poses,xs, ys, thetas, lidar_points):

    graph = gtsam.NonlinearFactorGraph()

    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(Vector3(0.3, 0.3, 0.1))
    graph.add(gtsam.PriorFactorPose2(0, Pose2(xs[0], ys[0], thetas[0]), priorNoise))

    nodeNoise = gtsam.noiseModel.Diagonal.Sigmas(Vector3(0.2, 0.2, 0.1))
    for i in tqdm(range(1,len(poses))):
        delta_pose = np.linalg.inv(poses[i-1])@poses[i]
        deltax, deltay, deltatheta = get_x_y_theta(delta_pose)
        graph.add(BetweenFactorPose2(i-1, i, Pose2(deltax, deltay, deltatheta), nodeNoise))
        if i-10>=0 and i%50==0:
            delta_pose = np.load(args.basedir + '/' + args.dataset + '/processed_lidar/fixed/pose_'+str(i-10)+'_'+str(i)+'.npy')
            deltax, deltay, deltatheta = get_x_y_theta(delta_pose)
            graph.add(BetweenFactorPose2(i-10, i, Pose2(deltax, deltay, deltatheta), nodeNoise))

    proximity_files = os.listdir('data/'+args.dataset+'/processed_lidar/proximity')
    for i in range(len(proximity_files)):
        if i%100==0:
            first_underscore = proximity_files[i].find('_')
            second_underscore = proximity_files[i].find('_', first_underscore + 1)
            src = int(proximity_files[i][first_underscore+1:second_underscore])
            dest = int(proximity_files[i][second_underscore+1:-4])
            delta_pose = np.load(args.basedir + '/' + args.dataset + '/processed_lidar/proximity/pose_'+str(src)+'_'+str(dest)+'.npy')
            deltax, deltay, deltatheta = get_x_y_theta(delta_pose)
            graph.add(BetweenFactorPose2(src, dest, Pose2(deltax, deltay, deltatheta), nodeNoise))

    poseEstimates = gtsam.Values()
    for i,pose in enumerate(poses):
        x,y,theta = get_x_y_theta(pose)
        poseEstimates.insert(i, gtsam.Pose2(0,0,0))

    parameters = gtsam.LevenbergMarquardtParams()
    parameters.setVerbosity("ERROR")
    parameters.setAbsoluteErrorTol(1e-6)
    parameters.setRelativeErrorTol(1e-6)
    parameters.setMaxIterations(100)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, poseEstimates, parameters)
    result = optimizer.optimize()
    Xs, Ys, thetas = [],[],[]
    for key in result.keys():
        pose = result.atPose2(key)
        x = pose.x()
        y = pose.y()
        theta = pose.theta()
        Xs.append(x)
        Ys.append(y)
        thetas.append(theta)
    plot_xytheta_poses(Xs,Ys,thetas, args.basedir+'/'+args.dataset+'/gtsam_poses_all'+args.dataset+'.pdf','gtsam poses for dataset '+args.dataset)
    print("done")
    return Xs,Ys,thetas