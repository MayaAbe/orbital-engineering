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

    # dvの値が最小値から5番目までの点を見つける
    indices = np.argsort(dv_values)[:50]

    # 通常のプロット
    scatter = ax.scatter(dv1_x_values, dv1_y_values, dv_values, c=time_cost_values, cmap='viridis', s=1)

    # 特定の点を赤く大きくプロット
    ax.scatter(dv1_x_values[indices], dv1_y_values[indices], dv_values[indices], c='red', s=50)

    ax.set_xlabel('Δv1_x (km/s)')
    ax.set_ylabel('Δv1_y (km/s)')
    ax.set_zlabel('Δv (km/s)')
    cbar = plt.colorbar(scatter, label='Time Cost (step)')
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))
    plt.title('Δv - Δv1_x - Δv1_y - Time Cost')

    # アスペクト比を同じに設定
    set_axes_equal(ax)

    # 軸のラベルを指数表記に設定
    ax.ticklabel_format(axis='x', style='sci')
    ax.ticklabel_format(axis='y', style='sci')
    ax.ticklabel_format(axis='z', style='sci')
    ax.xaxis.major.formatter._useMathText = True
    ax.yaxis.major.formatter._useMathText = True
    ax.zaxis.major.formatter._useMathText = True

    plt.legend()
    plt.show()

def plot_2d_timecost_vs_dv(dv_values, time_cost_values):
    plt.figure()
    plt.scatter(time_cost_values, dv_values, s=1)
    plt.xlabel('Time Cost (step)')
    plt.ylabel('Δv (km/s)')
    plt.title('2D plot of Δv vs Time Cost')
    plt.grid(True)

    # 軸のラベルを指数表記に設定
    plt.gca().ticklabel_format(axis='x', style='sci')
    plt.gca().ticklabel_format(axis='y', style='sci')
    plt.gca().xaxis.major.formatter._useMathText = True
    plt.gca().yaxis.major.formatter._useMathText = True

    plt.show()

# CSVファイルからデータを読み込み、指定した範囲でプロット
dv1_x_values, dv1_y_values, dv_values, time_cost_values = load_from_csv("gs_results.csv")
plot_3d_heatmap(dv1_x_values, dv1_y_values, dv_values, time_cost_values, dv1_x_range=(-5, 5), dv1_y_range=(-5, 5), dv_range=(0, 20))

# plot_2d_timecost_vs_dv(dv_values, time_cost_values)
