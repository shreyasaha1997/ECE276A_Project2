import open3d as o3d
from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
from .plot_util import *

def get_data_association(src_pts, target_pts, pose):
  z = target_pts
  R, t = pose[:3,:3], pose[:3,3]
  target_pts = [R@x+t for x in target_pts]
  kdtree = KDTree(src_pts)
  distances, indices = kdtree.query(target_pts, k=1)
  associated_src_pts = src_pts[indices]
  m = associated_src_pts
  return m,z

def calculate_R(zi, mi):
  zi = np.expand_dims(zi, axis=-1)
  mi = np.expand_dims(mi, axis=-1)
  Q = mi*zi.transpose(0,2,1)
  Q = np.sum(Q, axis=0)
  U, sigma, Vt = np.linalg.svd(Q)
  idet = np.eye(3)
  idet[2][2] = np.linalg.det(U@Vt)
  R = U@idet@Vt
  return R

def calculate_p(zi, mi, R):
  m_avg = sum(mi)/len(mi)
  z_avg = sum(zi)/len(zi)
  p = m_avg - R@z_avg
  return p

def implement_kabsch_algorithm(zi, mi):  
  R = calculate_R(zi, mi)
  p = calculate_p(zi, mi, R)
  return R, p

def icp(source_pc, target_pc, pose):
  for i in (range(20)):
    mi, zi = get_data_association(np.asarray(source_pc), np.asarray(target_pc), pose)
    R, p = implement_kabsch_algorithm(zi,mi)
    pose = np.hstack((R, p.reshape(-1, 1)))
    pose = np.vstack((pose, np.array([0,0,0,1])))
  return pose

def get_lidar_scan_pcds(args, poses, X, Y, lidar_points):  
    print(poses.shape, len(lidar_points))
    icp_poses = [poses[0]]
    for i in tqdm(range(1,len(lidar_points))):
      source_pts = lidar_points[i-1]
      target_pts = lidar_points[i]
      pose_src = poses[i-1]
      t_pose_t1 = np.linalg.inv(pose_src)@poses[i]
      t_pose_t1 = icp(source_pts, target_pts, t_pose_t1)
      icp_poses.append(icp_poses[i-1]@t_pose_t1)
      if i%50==0:
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(lidar_points[i-1].reshape(-1, 3))
        source_pcd.paint_uniform_color([0, 0, 1])

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(lidar_points[i].reshape(-1, 3))
        target_pcd.paint_uniform_color([1, 0, 0])
        target_pcd.transform(t_pose_t1)
        o3d.visualization.draw_geometries([source_pcd,target_pcd])
    plot_robot_poses(icp_poses[:], args.basedir + '/' + args.dataset + '/scan_matching_poses' + args.dataset + '.pdf', 'Robot Poses for dataset ' + args.dataset)
    Xs = [T[:3,3][0] for T in icp_poses]
    Ys = [T[:3,3][1] for T in icp_poses]
    R = [T[:3,:3] for T in icp_poses]
    thetas = [np.arctan2(r[1][0], r[0][0]) for r in R]
    return icp_poses, Xs, Ys, thetas