
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def get_initial_translation(src_pts, target_pts):
  x_src = sum(src_pts[:,0])/len(src_pts)
  y_src = sum(src_pts[:,1])/len(src_pts)
  z_src = sum(src_pts[:,2])/len(src_pts)

  x_tgt = sum(target_pts[:,0])/len(target_pts)
  y_tgt = sum(target_pts[:,1])/len(target_pts)
  z_tgt = sum(target_pts[:,2])/len(target_pts)

  translation = [x_src-x_tgt, y_src-y_tgt,z_src-z_tgt]
  return np.array(translation)

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
  for i in tqdm(range(40)):
    mi, zi = get_data_association(np.asarray(source_pc), np.asarray(target_pc), pose)
    R, p = implement_kabsch_algorithm(zi,mi)
    pose = np.hstack((R, p.reshape(-1, 1)))
    pose = np.vstack((pose, np.array([0,0,0,1])))
    # visualize_icp_result(source_pc, target_pc, pose)
  return pose

if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    # estimated_pose, you need to estimate the pose with ICP
    initial_translation = get_initial_translation(source_pc,target_pc)
    pose = np.eye(4)
    pose[:3,3] = initial_translation

    roll_angle_rad = np.radians(300)
    cos_roll = np.cos(roll_angle_rad)
    sin_roll = np.sin(roll_angle_rad)
    R = np.array([[cos_roll, -sin_roll, 0],
                     [sin_roll, cos_roll, 0],
                     [0, 0, 1]])
    # pose[:3,:3] = R

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
    
    target_pcd.transform(pose)
    
    # pose = np.eye(4)
    # pose[:3,:3] = R
    # target_pcd.transform(pose)
    # target_pc = np.asarray(target_pcd.points)  
    pose = icp(source_pc, target_pc, np.eye(4))


    visualize_icp_result(source_pc, target_pc, pose)
    print(1/0)
    
## kabsch algorithm works
## ICP works if the initialization is very good