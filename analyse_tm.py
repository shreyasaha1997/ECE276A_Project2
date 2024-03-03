import numpy as np
import cv2
import matplotlib.pyplot as plt

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

imd = cv2.imread('data/20/Disparity/disparity20_1.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
imc = cv2.imread('data/20/RGB/rgb20_1.png')[...,::-1]

disparity = imd.astype(np.float32)
  
# get depth
dd = (-0.00304 * disparity + 3.31)
z = 1.03 / dd

# calculate u and v coordinates 
v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
#u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

# get 3D coordinates 
fx = 585.05108211
fy = 585.05108211
cx = 315.83800193
cy = 242.94140713
x = (u-cx) / fx * z
y = (v-cy) / fy * z

# calculate the location of each pixel in the RGB image
rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
rgbv = np.round((v * 526.37 + 16662.0)/fy)

valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
x,y,z,rgbu,rgbv = x[valid],y[valid],z[valid],rgbu[valid],rgbv[valid]

xyz = np.array([[a,b,c] for a,b,c in zip(x,y,z)])
oRr = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
xyz = np.array([np.linalg.inv(oRr)@X for X in xyz])



bRr = rotation_matrix(0,  0.36,  0.021)
btr = np.array([0.18, 0.005, 0.36])
bTr = np.eye(4)
bTr[:3, :3] = bRr
bTr[:3, 3] = btr
xyz = np.array([(bRr@X + btr) for X in xyz])

x,y,z = xyz[:,0], xyz[:,1], xyz[:,2]

valid2 = (z<0.3)

# display valid RGB pixels
fig = plt.figure(figsize=(10, 13.3))
# ax = fig.add_subplot(projection='3d')
plt.scatter(x[valid2],y[valid2],c=imc[rgbv[valid2].astype(int),rgbu[valid2].astype(int)]/255.0)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.view_init(elev=0, azim=180)
plt.savefig('ref.png')