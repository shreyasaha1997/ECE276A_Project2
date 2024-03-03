import numpy as np
import configargparse
import os
from utils.dataloader import *
from utils.ododmetry import *
from utils.lidar_utils import *
from utils.texture_mapping import *
from utils.factor_graph import *
from utils.occupancy_grid import *
import matplotlib.pyplot as plt 

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='data/', 
                        help='where to load the data from')
    parser.add_argument("--dataset", type=str, default='20', 
                        help='experiment name')
    parser.add_argument("--epochs", type=int, default='600', 
                        help='number of epochs')
    parser.add_argument("--icp_refinement", action='store_true', 
                        help='icp_refinement')
    parser.add_argument("--create_texture_map", action='store_true', 
                        help='create_texture_map')
    parser.add_argument("--gtsam_refinement", action='store_true', 
                        help='gtsam_refinement')
    parser.add_argument("--plot_comparisons", action='store_true', 
                        help='plot_comparisons')
    parser.add_argument("--occupancy_grid", action='store_true', 
                        help='occupancy_grid')
    parser.add_argument("--motion_model", action='store_true', 
                        help='occumotion_modelpancy_grid')
    return parser


if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    vt, yaws, ts, lidar_points, rgbs, disps, rgbs_ts_indices, disps_ts_indices = load_data(args)

    if args.motion_model:
        poses, Xs, Ys, thetas = initialize_robot_pose(args, vt, yaws, ts)
        np.save(args.basedir + '/' + args.dataset + '/poses/initialised_poses_motion_model.npy',poses)
        np.save(args.basedir + '/' + args.dataset + '/poses/X_motion_model.npy',Xs)
        np.save(args.basedir + '/' + args.dataset + '/poses/Y_motion_model.npy',Ys)
        np.save(args.basedir + '/' + args.dataset + '/poses/theta_motion_model.npy',thetas)

    if args.icp_refinement:
        poses = np.load(args.basedir + '/' + args.dataset + '/poses/initialised_poses_motion_model.npy')
        X = np.load(args.basedir + '/' + args.dataset + '/poses/X_motion_model.npy')
        Y = np.load(args.basedir + '/' + args.dataset + '/poses/Y_motion_model.npy')
        poses, Xs, Ys, thetas = get_lidar_scan_pcds(args, poses, X, Y, lidar_points)
        np.save(args.basedir + '/' + args.dataset + '/poses/initialised_poses_icpo.npy',poses)
        np.save(args.basedir + '/' + args.dataset + '/poses/X_icp.npy',Xs)
        np.save(args.basedir + '/' + args.dataset + '/poses/Y_icp.npy',Ys)
        np.save(args.basedir + '/' + args.dataset + '/poses/theta_icp.npy',thetas)

    if args.occupancy_grid:
        Xs = np.load(args.basedir + '/' + args.dataset + '/poses/X_gtsam.npy')
        Ys = np.load(args.basedir + '/' + args.dataset + '/poses/Y_gtsam.npy')
        thetas = np.load(args.basedir + '/' + args.dataset + '/poses/theta_gtsam.npy')
        create_occupancy_grid_map(args, Xs, Ys, thetas, lidar_points)
    
    if args.gtsam_refinement:
        poses = np.load(args.basedir + '/' + args.dataset + '/poses/initialised_poses_motion_model.npy')
        X = np.load(args.basedir + '/' + args.dataset + '/poses/X_motion_model.npy')
        Y = np.load(args.basedir + '/' + args.dataset + '/poses/Y_motion_model.npy')
        thetas = np.load(args.basedir + '/' + args.dataset + '/poses/theta_motion_model.npy')
        Xs, Ys, thetas = create_factor_graph(args, poses[:], X[:], Y[:], thetas[:],lidar_points[:])
        np.save(args.basedir + '/' + args.dataset + '/poses/X_gtsam.npy',Xs)
        np.save(args.basedir + '/' + args.dataset + '/poses/Y_gtsam.npy',Ys)
        np.save(args.basedir + '/' + args.dataset + '/poses/theta_gtsam.npy',thetas)

    if args.create_texture_map:
        poses = np.load(args.basedir + '/' + args.dataset + '/poses/initialised_poses_motion_model.npy')
        Xs = np.load(args.basedir + '/' + args.dataset + '/poses/X_motion_model.npy')
        Ys = np.load(args.basedir + '/' + args.dataset + '/poses/Y_motion_model.npy')
        thetas = np.load(args.basedir + '/' + args.dataset + '/poses/theta_motion_model.npy')
        create_texture_map(args,rgbs, disps, poses, disps_ts_indices, args.basedir + '/' + args.dataset +'/texture_map_'+str(args.dataset)+'.png')
        print("done")

    if args.plot_comparisons:
        X1 = np.load(args.basedir + '/' + args.dataset + '/poses/X_motion_model.npy')
        Y1 = np.load(args.basedir + '/' + args.dataset + '/poses/Y_motion_model.npy')
        X2 = np.load(args.basedir + '/' + args.dataset + '/poses/X_icp.npy')
        Y2 = np.load(args.basedir + '/' + args.dataset + '/poses/Y_icp.npy')
        X3 = np.load(args.basedir + '/' + args.dataset + '/poses/X_gtsam.npy')
        Y3 = np.load(args.basedir + '/' + args.dataset + '/poses/Y_gtsam.npy')
        plot_comparisons(X1, Y1, 'Motion Model Odometry', X2, Y2, 'Scan Matching Via ICP', X3, Y3, 'GTSAM Refinement Via ICP', 'Robot Trajectory Coparison for dataset ' + args.dataset, args.basedir + '/' + args.dataset + '/comparison.pdf')


