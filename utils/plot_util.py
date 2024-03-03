import numpy as np
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def plot_robot_poses_orientations(x_vals, y_vals, theta_vals, title):
    # Plot each robot pose and orientation
    # for x, y, theta in zip(x_vals, y_vals, theta_vals):
        # Plot robot pose
    plt.scatter(x_vals, y_vals, c=np.arange(len(x_vals)), cmap='viridis', marker='o', s=10)  # Plot robot position as a red dot
    plt.colorbar(label='Timestamps')

    # Set plot limits
    plt.xlim([np.min(x_vals)-10, np.max(x_vals)+10])  # Set x-axis limits
    plt.ylim([np.min(y_vals)-10, np.max(y_vals)+10])  # Set y-axis limits

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    # Add grid
    plt.grid(True)
    return plt

def plot_robot_poses(poses, path, title):
    x = [p[0][3] for p in poses]
    y = [p[1][3] for p in poses]
    Rts = [p[:3,:3] for p in poses]

    print(min(x), max(x),min(y),max(y))
    theta = [np.arctan2(Rt[1, 0], Rt[0, 0]) for Rt in Rts]
    plt = plot_robot_poses_orientations(x, y, theta, title)
    plt.savefig(path)
    plt.close()

def plot_xytheta_poses(Xs,Ys,thetas, path, title):
    plt = plot_robot_poses_orientations(Xs, Ys, thetas, title)
    plt.savefig(path)

def plot_occupancy_map(empty_map, rx, ry, path, title):
    X, Y = [],[]
    for i in range(1201):
      for j in range(1201):
        X.append(i)
        Y.append(j)
    colors = np.array([empty_map['map'][x][y] for x,y in zip(X, Y)])
    plt.scatter(X, Y, c=colors, cmap='viridis_r', marker='o', s=20)
    plt.scatter(rx, ry, s=2, c='r')
    plt.colorbar(label='Probability of cell to be occupied')
    plt.title(title)
    plt.savefig(path)

def plot_comparisons(X1, Y1, tlabel1, X2, Y2, tlabel2, X3, Y3, tlabel3, title, path):
   plt.figure(figsize=(12, 10))
   plt.scatter(X1, Y1, c=np.arange(len(X1)), cmap='viridis', marker='o', s=10)  # Plot robot position as a red dot
   plt.colorbar(label=tlabel1)
   plt.scatter(X2, Y2, c=np.arange(len(X2)), cmap='inferno', marker='o', s=10)  # Plot robot position as a red dot
   plt.colorbar(label=tlabel2)
   plt.scatter(X3, Y3, c=np.arange(len(X3)), cmap='twilight', marker='o', s=10)  # Plot robot position as a red dot
   plt.colorbar(label=tlabel3)
   plt.grid(True)
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title(title)
   plt.savefig(path)