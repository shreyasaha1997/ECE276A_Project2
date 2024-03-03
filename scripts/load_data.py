import numpy as np


if __name__ == '__main__':
  dataset = 21
  
  # with np.load("data/Encoders%d.npz"%dataset) as data:
  #   encoder_counts = data["counts"] # 4 x n encoder counts
  #   encoder_stamps = data["time_stamps"] # encoder time stamps

  # with np.load("data/Hokuyo%d.npz"%dataset) as data:
  #   lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
  #   lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
  #   lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
  #   lidar_range_min = data["range_min"] # minimum range value [m]
  #   lidar_range_max = data["range_max"] # maximum range value [m]
  #   lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
  #   lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
    
  # with np.load("data/Imu%d.npz"%dataset) as data:
  #   imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
  #   imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
  #   imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  # print(encoder_counts.shape, encoder_stamps.shape, imu_angular_velocity.shape, imu_linear_acceleration.shape, imu_stamps.shape)

  # np.save('data/' + str(dataset) + '/enc.npy', encoder_counts)
  # np.save('data/' + str(dataset) + '/enc_ts.npy', encoder_stamps)
  # np.save('data/' + str(dataset) + '/imu_w.npy', imu_angular_velocity)
  # np.save('data/' + str(dataset) + '/imu_a.npy', imu_linear_acceleration)
  # np.save('data/' + str(dataset) + '/imu_ts.npy', imu_stamps)

  # np.save('data/' + str(dataset) + '/lidar_angle_min.npy', lidar_angle_min)
  # np.save('data/' + str(dataset) + '/lidar_angle_max.npy', lidar_angle_max)
  # np.save('data/' + str(dataset) + '/lidar_angle_increment.npy', lidar_angle_increment)
  # np.save('data/' + str(dataset) + '/lidar_range_min.npy', lidar_range_min)
  # np.save('data/' + str(dataset) + '/lidar_range_max.npy', lidar_range_max)
  # np.save('data/' + str(dataset) + '/lidar_ranges.npy', lidar_ranges)
  # np.save('data/' + str(dataset) + '/lidar_ts.npy', lidar_stamsp)

  # print(lidar_angle_min.shape, lidar_angle_max.shape, lidar_angle_increment.shape, lidar_range_min.shape, lidar_range_max.shape, lidar_ranges.shape, lidar_stamsp.shape)


  with np.load("data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    
  np.save('data/' + str(dataset) + '/disp_stamps.npy', disp_stamps)
  np.save('data/' + str(dataset) + '/rgb_stamps.npy', rgb_stamps)