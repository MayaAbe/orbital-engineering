import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_from_csv(filename):
    data = pd.read_csv(filename)
    dv1_x_values = data["dv1_x"].values
    dv1_y_values = data["dv1_y"].values
    dv_values = data["dv"].values
    time_cost_values = data["time_cost"].values
    return dv1_x_values, dv1_y_values, dv_values, time_cost_values

def set_axes_equal(ax):
    """Make the 3D plot axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def plot_3d_heatmap(dv1_x_values, dv1_y_values, dv_values, time_cost_values, dv1_x_range=None, dv1_y_range=None, dv_range=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 範囲を指定
    if dv1_x_range is not None:
        mask = (dv1_x_values >= dv1_x_range[0]) & (dv1_x_values <= dv1_x_range[1])
        dv1_x_values = dv1_x_values[mask]
        dv1_y_values = dv1_y_values[mask]
        dv_values = dv_values[mask]
        time_cost_values = time_cost_values[mask]

    if dv1_y_range is not None:
        mask = (dv1_y_values >= dv1_y_range[0]) & (dv1_y_values <= dv1_y_range[1])
        dv1_x_values = dv1_x_values[mask]
        dv1_y_values = dv1_y_values[mask]
        dv_values = dv_values[mask]
        time_cost_values = time_cost_values[mask]

    if dv_range is not None:
        mask = (dv_values >= dv_range[0]) & (dv_values <= dv_range[1])
        dv1_x_values = dv1_x_values[mask]
        dv1_y_values = dv1_y_values[mask]
        dv_values = dv_values[mask]
        time_cost_values = time_cost_values[mask]

    scatter = ax.scatter(dv1_x_values, dv1_y_values, dv_values, c=time_cost_values, cmap='viridis', s=1)
    ax.set_xlabel('dv1_x_value')
    ax.set_ylabel('dv1_y_value')
    ax.set_zlabel('dv')
    plt.colorbar(scatter, label='time_cost')
    plt.title('3D plot of dv vs dv1_x_value and dv1_y_value with time_cost')

    # アスペクト比を同じに設定
    set_axes_equal(ax)

    plt.show()

def plot_2d_slice(dv1_x_values, dv1_y_values, dv_values, time_cost_values, dv1_y_target):
    mask = np.isclose(dv1_y_values, dv1_y_target)
    filtered_dv1_x_values = dv1_x_values[mask]
    filtered_dv_values = dv_values[mask]
    filtered_time_cost_values = time_cost_values[mask]

    plt.figure()
    scatter = plt.scatter(filtered_dv1_x_values, filtered_dv_values, c=filtered_time_cost_values, cmap='viridis', s=1)
    plt.colorbar(scatter, label='time_cost')
    plt.xlabel('dv1_x_value')
    plt.ylabel('dv')
    plt.title(f'2D plot of dv vs dv1_x_value with time_cost (dv1_y={dv1_y_target:.2f})')
    plt.grid(True)
    plt.show()

# CSVファイルからデータを読み込み、指定した範囲でプロット
filename = "gs_results.csv"
dv1_x_values, dv1_y_values, dv_values, time_cost_values = load_from_csv(filename)

# 3Dヒートマップのプロット
plot_3d_heatmap(dv1_x_values, dv1_y_values, dv_values, time_cost_values, dv1_x_range=(-10, 10), dv1_y_range=(-10, 10), dv_range=(0, 20))

# dv1_y=-3.0900000000000407の時の2Dスライスプロット
plot_2d_slice(dv1_x_values, dv1_y_values, dv_values, time_cost_values, dv1_y_target=-3.080000000000041)
