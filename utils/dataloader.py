import os
import numpy as np
import pickle
import imageio

def calculate_linear_velocity(enc, enc_ts):
    fr, fl, rr, rl = enc[0], enc[1], enc[2], enc[3]
    rd = 0.5 * 0.0022 * (fr + rr)
    ld = 0.5 * 0.0022 * (fl + rl)
    td = 0.5 * (rd + ld)
    vt = [0.]
    for i in range(1,len(enc_ts)):
        v = (td[i])/(enc_ts[i] - enc_ts[i-1])
        vt.append(v)
    return np.array(vt)

def choose_yaw_close_to_enc_ts(imu_ts, enc_ts):
    closest_indices = []
    for t in enc_ts:
        closest_index = np.argmin(np.abs(imu_ts - t))
        closest_indices.append(closest_index)
    return closest_indices

def process_individual_lidar_data(lidar_ranges, min_angle, max_angle, angle_increment, min_range, max_range):
    lidar_angles = np.arange(min_angle,max_angle+(angle_increment*0.01),angle_increment)
    valid_indices = np.logical_and((lidar_ranges < max_range),(lidar_ranges> min_range))
    ranges = lidar_ranges[valid_indices]
    angles = lidar_angles[valid_indices]
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    robot_R_lidar = np.eye(3)
    robot_t_lidar = np.array([0.3,0,0])
    lidar_pts = [(robot_R_lidar@np.array([x,y,0]) + robot_t_lidar) for x,y in zip(xs0,ys0)]
    return np.array(lidar_pts)

def process_lidar_data(lidar_ranges, min_angle, max_angle, angle_increment, min_range, max_range):
    lidar_ranges = lidar_ranges.transpose()
    lidar_points = [process_individual_lidar_data(l, min_angle, max_angle, angle_increment, min_range, max_range) for l in lidar_ranges]
    return lidar_points

def choose_lidar_ts_close_to_enc_ts(imu_ts, lidar_ts):
    closest_indices = []
    for t in lidar_ts:
        closest_index = np.argmin(np.abs(imu_ts - t))
        closest_indices.append(closest_index)
    return closest_indices

def choose_disps_close_to_rgbs(rgb_ts, disp_ts, disps):
    closest_indices = []
    for t in rgb_ts:
        closest_index = np.argmin(np.abs(disp_ts - t))
        closest_indices.append(closest_index)
    return disps[closest_indices], disp_ts[closest_indices]

def choose_rgbs_disps_close_to_imus(image_ts, imu_ts):
    closest_indices = []
    for t in image_ts:
        closest_index = np.argmin(np.abs(imu_ts - t))
        closest_indices.append(closest_index)
    return np.array(closest_indices)

def load_data(args):
    basedir = os.path.join(args.basedir, args.dataset)
    processed_dir = os.path.join(basedir, 'processed_dir')

    if os.path.exists(processed_dir):
        vt = np.load(os.path.join(processed_dir, 'vt.npy')).astype(float)
        yaws = np.load(os.path.join(processed_dir, 'yaws.npy')).astype(float) 
        imu_ts = np.load(os.path.join(processed_dir, 'imu_ts.npy')).astype(float) 
        with open(processed_dir + '/disps.pkl', "rb") as file1:
            disps = pickle.load(file1)
        with open(processed_dir + '/rgbs.pkl', "rb") as file2:
            rgbs = pickle.load(file2)
        with open(processed_dir + '/rgbs_ts_indices.pkl', "rb") as file2:
            rgbs_ts_indices = pickle.load(file2)
        with open(processed_dir + '/disps_ts_indices.pkl', "rb") as file2:
            disps_ts_indices = pickle.load(file2)
        with open(processed_dir + '/lidar_points.pkl', "rb") as file:
            lidar_points = pickle.load(file)
        return vt, yaws, imu_ts, lidar_points, np.array(rgbs), np.array(disps), np.array(rgbs_ts_indices), np.array(disps_ts_indices)

    enc_ts = np.load(os.path.join(basedir, 'inputs/enc_ts.npy')).astype(float)
    enc = np.load(os.path.join(basedir, 'inputs/enc.npy')).astype(float)
    accl = np.load(os.path.join(basedir, 'inputs/imu_a.npy')).astype(float)
    omegas = np.load(os.path.join(basedir, 'inputs/imu_w.npy')).astype(float)
    imu_ts = np.load(os.path.join(basedir, 'inputs/imu_ts.npy')).astype(float)

    lidar_ranges = np.load(os.path.join(basedir, 'inputs/lidar_ranges.npy')).astype(float)
    lidar_ts = np.load(os.path.join(basedir, 'inputs/lidar_ts.npy')).astype(float)
    lidar_max_angle = np.load(os.path.join(basedir, 'inputs/lidar_angle_max.npy')).astype(float)
    lidar_min_angle = np.load(os.path.join(basedir, 'inputs/lidar_angle_min.npy')).astype(float)
    lidar_max_range = np.load(os.path.join(basedir, 'inputs/lidar_range_max.npy')).astype(float)
    lidar_min_range = np.load(os.path.join(basedir, 'inputs/lidar_range_min.npy')).astype(float)
    lidar_angle_increment = np.load(os.path.join(basedir, 'inputs/lidar_angle_increment.npy')).astype(float)
    lidar_points = process_lidar_data(lidar_ranges, lidar_min_angle, lidar_max_angle, lidar_angle_increment, lidar_min_range, lidar_max_range)

    yaws = omegas[2]
    vt = calculate_linear_velocity(enc,enc_ts)

    closest_yaw_indices = choose_yaw_close_to_enc_ts(imu_ts, enc_ts)
    yaws = yaws[closest_yaw_indices]
    imu_ts = imu_ts[closest_yaw_indices]

    closest_lidar_indices = choose_lidar_ts_close_to_enc_ts(lidar_ts, imu_ts)
    lidar_ts = lidar_ts[closest_lidar_indices]
    lidar_points = [lidar_points[i] for i in closest_lidar_indices]

    disp_ts = np.load(os.path.join(basedir, 'inputs/disp_stamps.npy')).astype(float)
    rgb_ts = np.load(os.path.join(basedir, 'inputs/rgb_stamps.npy')).astype(float)
    rgbfiles = os.listdir(os.path.join(basedir, 'RGB'))
    dispfiles = os.listdir(os.path.join(basedir, 'Disparity'))
    rgbs = np.array([imageio.imread(os.path.join(basedir,'RGB/'+f)) for f in rgbfiles])
    disps = np.array([imageio.imread(os.path.join(basedir,'Disparity/'+f)) for f in dispfiles])
    disps, disp_ts = choose_disps_close_to_rgbs(rgb_ts, disp_ts, disps)
    rgbs_ts_indices = choose_rgbs_disps_close_to_imus(rgb_ts, imu_ts)
    disps_ts_indices = choose_rgbs_disps_close_to_imus(disp_ts, imu_ts)
    os.makedirs(processed_dir)
    np.save(processed_dir + '/vt.npy', vt)
    np.save(processed_dir + '/yaws.npy', yaws)
    np.save(processed_dir + '/imu_ts.npy', imu_ts)
    with open(processed_dir + '/lidar_points.pkl', "wb") as file:
        pickle.dump(lidar_points, file)
    with open(processed_dir + '/rgbs.pkl', "wb") as file:
        pickle.dump(rgbs, file)
    with open(processed_dir + '/disps.pkl', "wb") as file:
        pickle.dump(disps, file)
    with open(processed_dir + '/rgbs_ts_indices.pkl', "wb") as file:
        pickle.dump(rgbs_ts_indices, file)
    with open(processed_dir + '/disps_ts_indices.pkl', "wb") as file:
        pickle.dump(disps_ts_indices, file)
    return vt, yaws, imu_ts, lidar_points, rgbs, disps, rgbs_ts_indices, disps_ts_indices