import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .plot_util import *

def create_empty_map():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -30  #meters
    MAP['ymin']  = -30
    MAP['xmax']  =  30
    MAP['ymax']  =  30 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=float)
    return MAP

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return x,y

def create_occupancy_map_for_single_lidar(empty_map, rx, ry, xs0, ys0):
    # Y = np.stack((xs0,ys0))
    xis = np.ceil((xs0 - empty_map['xmin']) / empty_map['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - empty_map['ymin']) / empty_map['res'] ).astype(np.int16)-1

    r0_x = np.ceil((rx - empty_map['xmin']) / empty_map['res'] ).astype(np.int16)-1
    r0_y = np.ceil((ry - empty_map['ymin']) / empty_map['res'] ).astype(np.int16)-1

    free_cells, occ_cells = [],[]
    nz_x, nz_y = [],[]

    for xi,yi in zip(xis, yis):
        xf, yf = bresenham2D(r0_x, r0_y, xi, yi)
        for x,y in zip(xf,yf):
          if x==xi and y==yi:
            continue
          free_cells.append((int(x),int(y)))
        occ_cells.append((int(xi),int(yi)))
    
    free_cells = set(free_cells)
    occ_cells = set(occ_cells)

    for (x,y) in free_cells:
      empty_map['map'][x][y] = empty_map['map'][x][y] - np.log(4)
      nz_x.append(x)
      nz_y.append(y)
    for (x,y) in occ_cells:
      empty_map['map'][x][y] = empty_map['map'][x][y] + np.log(4)
      nz_x.append(x)
      nz_y.append(y)
        
    return empty_map, r0_x, r0_y, xis, yis,nz_x, nz_y

def get_occupancy_from_logodds(empty_map):
  for i in range(len(empty_map)):
    for j in range(len(empty_map)):
    #   print(i,j, np.exp(empty_map[i][j]))
      empty_map[i][j] = 1/((1/np.exp(empty_map[i][j]))+1)
  return empty_map
  

def create_occupancy_grid_map(args, Xs, Ys, thetas, lidar_points):
    empty_map = create_empty_map()
    rx, ry = [],[]
    i=1
    for x, y, theta, lidar_point in tqdm(zip(Xs, Ys, thetas, lidar_points)):
      lidar_point = lidar_point[:,:2]
      R, t = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]]), np.array([x,y])
      transformed_lidar_point = np.array([np.matmul(R,l)+t for l in lidar_point])
      empty_map, r0_x, r0_y, xl, yl,nz_x, nz_y = create_occupancy_map_for_single_lidar(empty_map,x,y, transformed_lidar_point[:,0], transformed_lidar_point[:,1])
      rx.append(r0_x)
      ry.append(r0_y)
    empty_map['map'] = get_occupancy_from_logodds(empty_map['map'])

    plot_occupancy_map(empty_map, rx, ry, args.basedir + '/' + args.dataset + '/occupancy_map_gtsam_'+args.dataset+'.png', "Occupancy grid map for dataset " + args.dataset)
    return