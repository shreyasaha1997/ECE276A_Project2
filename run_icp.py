import numpy as np
from utils.lidar_utils import *
import pickle
from tqdm import tqdm

def pose_distances(pose1, pose2):
    R1 = pose1[:3, :3]
    t1 = pose1[:3, 3]
    R2 = pose2[:3, :3]
    t2 = pose2[:3, 3]

    R_rel = np.dot(R2, R1.T)
    rot_dist = np.arccos((np.trace(R_rel) - 1) / 2)
    t_rel = t2 - np.dot(R_rel, t1)

    trans_dist = np.linalg.norm(t_rel)
    return rot_dist, trans_dist

dataset = '20'

poses = np.load('data/'+dataset+'/poses/initialised_poses_icpo.npy')
with open('data/'+dataset+'/processed_dir/lidar_points.pkl', "rb") as file:
    lidar_points = pickle.load(file)

# for i in range(len(poses)):
#     for j in range(i+1,len(poses)):
#         a,b = pose_distances(poses[i], poses[j])
#         if i!=j and j-i>10 and b<0.02 and a<0.003:
#             source_pts = lidar_points[i]
#             target_pts = lidar_points[j]
#             delta_pose = np.linalg.inv(poses[i])@poses[j]
#             delta_pose = icp(source_pts, target_pts, delta_pose)
#             np.save('data/'+dataset+'/processed_lidar/proximity/pose_'+str(i)+'_'+str(j)+'.npy', delta_pose)
            
            


for i in tqdm(range(10,len(lidar_points))):
    source_pts = lidar_points[i-10]
    target_pts = lidar_points[i]
    delta_pose = np.linalg.inv(poses[i-10])@poses[i]
    delta_pose = icp(source_pts, target_pts, delta_pose)
    np.save('data/'+dataset+'/processed_lidar/fixed/pose_'+str(i-10)+'_'+str(i)+'.npy', delta_pose)