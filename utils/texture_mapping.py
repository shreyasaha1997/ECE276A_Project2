import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d
from tqdm import tqdm
import pickle
import os

def create_empty_map():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -10  #meters
    MAP['ymin']  = -10
    MAP['xmax']  =  30
    MAP['ymax']  =  30 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    print(MAP['sizex'] , MAP['sizey'] )
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype=float)
    return MAP

def rotation_matrix(roll, pitch, yaw):
    """
    Create a 3D rotation matrix from roll, pitch, and yaw angles (in radians).
    """
    # Roll, pitch, and yaw rotation matrices
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    # Combine the rotation matrices
    rotation_matrix = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    
    return rotation_matrix

def correlate_disps_rgbs(rgb, disp, pose):
    ##step 1 and 2
    dd = (-0.00304 * disp + 3.31)
    z = 1.03 / dd
    v,u = np.mgrid[0:disp.shape[0],0:disp.shape[1]]
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disp.shape[1])&(rgbv>=0)&(rgbv<disp.shape[0])
    x,y,z,rgbu,rgbv = x[valid],y[valid],z[valid],rgbu[valid],rgbv[valid]
    xyz = np.array([[a,b,c] for a,b,c in zip(x,y,z)])

    ##step 3
    oRr = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    xyz = np.array([np.linalg.inv(oRr)@X for X in xyz])

    ##step 4
    bRr = rotation_matrix(0,  0.36,  0.021)
    btr = np.array([0.18, 0.005, 0.36])
    bTr = np.eye(4)
    bTr[:3, :3] = bRr
    bTr[:3, 3] = btr
    xyz = np.array([(bRr@X + btr) for X in xyz])

    x,y,z = xyz[:,0], xyz[:,1], xyz[:,2]
    valid2 = (z<0)
    
    ##step 5
    R = pose[:3,:3]
    t = pose[:3,3]
    xyz = np.array([R@X + t for X in xyz])

    return xyz[valid2], rgb[rgbv[valid2].astype(int),rgbu[valid2].astype(int),:]

def map_3d_points_to_cells(points3d, empty_map, colors):
    Xs, Ys, Zs = np.array(points3d[:,0]), np.array(points3d[:,1]), np.array(points3d[:,2])
    cxs = np.ceil((Xs - empty_map['xmin']) / empty_map['res'] ).astype(np.int16)-1
    cys = np.ceil((Ys - empty_map['ymin']) / empty_map['res'] ).astype(np.int16)-1
    mean_value = np.mean(Zs)
    min_value = np.min(Zs)
    max_value = np.max(Zs)
    std_deviation = np.std(Zs)
    # print("llll", mean_value, min_value, max_value, std_deviation)
    for x,y,z,c in zip(cxs,cys,Zs,colors):
        if z>0:
            continue
        if x>=801 or y>=801:
            continue
        
        if empty_map['map'][x][y][0] == 0 and empty_map['map'][x][y][1] == 0 and empty_map['map'][x][y][2] == 0:
            empty_map['map'][x][y] = c
    return empty_map

def plot_texture_map(empty_map, path, poses, title):
    X, Y = [],[]
    for i in range(801):
      for j in range(801):
        X.append(i)
        Y.append(j)
    colors = np.array([empty_map['map'][x][y] for x,y in zip(X, Y)])
    plt.scatter(X, Y, c=colors/255)
    Xs = np.array([p[:3,3][0] for p in poses])
    Ys = np.array([p[:3,3][1] for p in poses])
    Xs = np.ceil((Xs - empty_map['xmin']) / empty_map['res'] ).astype(np.int16)-1
    Ys = np.ceil((Ys - empty_map['ymin']) / empty_map['res'] ).astype(np.int16)-1
    # plt.scatter(Xs,Ys,s=1)

    # plt.scatter(Xs[:10],Ys[:10],s=3)
    plt.title(title)
    plt.savefig(path)

def create_texture_map(args,rgbs, disps, poses, indices,path):
    total_points = None
    total_colors = None
    empty_map = create_empty_map()
    i=-1
    for rgb, disp, ind in tqdm(zip(rgbs[:], disps[:], indices[:])):
        i=i+1
        if i<=300 and i%200!=0:
            continue
        if i>2000 and i%200!=0:
            continue
        if i%30!=0:
            continue
        if os.path.exists('data/'+args.dataset+'/processed_point_clouds/points3d_'+str(i)+'.npy'):
            points3d = np.load('data/'+args.dataset+'/processed_point_clouds/points3d_'+str(i)+'.npy')
            colors = np.load('data/'+args.dataset+'/processed_point_clouds/colors_'+str(i)+'.npy')
            empty_map = map_3d_points_to_cells(points3d, empty_map, colors)
            continue
        pose = poses[ind]
        points3d, colors = correlate_disps_rgbs(rgb, disp, pose)
        np.save('data/'+args.dataset+'/processed_point_clouds/points3d_'+str(i)+'.npy',points3d)
        np.save('data/'+args.dataset+'/processed_point_clouds/colors_'+str(i)+'.npy',colors)
        empty_map = map_3d_points_to_cells(points3d, empty_map, colors)
    plot_texture_map(empty_map, path, poses, 'texture mapping for dataset '+str(args.dataset))
    return