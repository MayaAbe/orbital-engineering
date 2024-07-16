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

    scatter = ax.scatter(dv1_x_values, dv1_y_values, dv_values, c=time_cost_values, cmap='viridis', s=10)
    ax.set_xlabel('dv1_x_value')
    ax.set_ylabel('dv1_y_value')
    ax.set_zlabel('dv')
    plt.colorbar(scatter, label='time_cost')
    plt.title('3D plot of dv vs dv1_x_value and dv1_y_value with time_cost')
    plt.show()

# CSVファイルからデータを読み込み、指定した範囲でプロット
dv1_x_values, dv1_y_values, dv_values, time_cost_values = load_from_csv("grid_search_results.csv")
plot_3d_heatmap(dv1_x_values, dv1_y_values, dv_values, time_cost_values, dv1_x_range=(-10, 10), dv1_y_range=(-3.5, 0.0), dv_range=(0, 10))
