import numpy as np
from .plot_util import *
from tqdm import tqdm

def initialize_robot_pose(args, vts, yaws, ts):
    poses = np.array([np.eye(4) for t in ts])
    Xs, Ys, thetas = [0.],[0.],[0.]
    for i in range(1, len(ts)):
        xt = Xs[i-1]
        yt = Ys[i-1]
        thetat = thetas[i-1]

        vt = vts[i]
        wt = yaws[i]
        taut = ts[i] - ts[i-1]

        xt1 = xt + vt*np.cos(thetat)*taut
        yt1 = yt + vt*np.sin(thetat)*taut
        thetat1 = thetat + wt*taut

        Xs.append(xt1)
        Ys.append(yt1)
        thetas.append(thetat1)
        poses[i][0][3] = xt1
        poses[i][1][3] = yt1
        poses[i][:3,:3] = np.array([[np.cos(thetat1), -np.sin(thetat1), 0],
                                [np.sin(thetat1), np.cos(thetat1), 0],
                                [0, 0, 1]])
        
    plot_xytheta_poses(Xs, Ys, thetas, args.basedir + '/' + args.dataset + '/initial_poses_'+args.dataset+'.pdf', 'Robot Poses for dataset ' + args.dataset)
    return poses, Xs, Ys, thetas

