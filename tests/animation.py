import core.two_body as tb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation

def create_orbit_animation(x1, y1, start_fraction=0.4, end_fraction=1.0, step=1000, interval=10, highlight_latest=True, speed_factor=1.0):
    solx, soly = tb.MoonEarthSat(x1, y1, 6, 10)

    # 表示するフレームの範囲を設定
    start_index = int(len(solx) * start_fraction)
    end_index = int(len(solx) * end_fraction)
    solx = solx[start_index:end_index]
    soly = soly[start_index:end_index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 軌道の範囲を設定
    ax.set_xlim([min(solx[:, 0]), max(solx[:, 0])])
    ax.set_ylim([min(solx[:, 1]), max(solx[:, 1])])
    ax.set_zlim([min(solx[:, 2]), max(solx[:, 2])])

    # 軸のアスペクト比を同じに設定するための関数
    def set_aspect_equal(ax):
        extents = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        ax.set_xlim3d([centers[0] - r, centers[0] + r])
        ax.set_ylim3d([centers[1] - r, centers[1] + r])
        ax.set_zlim3d([centers[2] - r, centers[2] + r])

    set_aspect_equal(ax)

    # 原点に大きな点をプロット
    ax.scatter(0, 0, 0, color='k', s=100)  # sは点のサイズを指定

    frames = []
    for i in range(0, len(solx), step):
        line_solx, = ax.plot(solx[:i, 0], solx[:i, 1], solx[:i, 2], 'b')
        line_soly, = ax.plot(soly[:i, 0], soly[:i, 1], soly[:i, 2], 'r')

        if highlight_latest:
            if i > 0:
                current_solx, = ax.plot([solx[i-1, 0]], [solx[i-1, 1]], [solx[i-1, 2]], 'bo', markersize=8)
                current_soly, = ax.plot([soly[i-1, 0]], [soly[i-1, 1]], [soly[i-1, 2]], 'ro', markersize=8)
                frames.append([line_solx, line_soly, current_solx, current_soly])
            else:
                frames.append([line_solx, line_soly])
        else:
            frames.append([line_solx, line_soly])

    ani = ArtistAnimation(fig, frames, interval=interval * speed_factor, blit=True)

    plt.grid()  # 格子をつける
    plt.show()

# 使用例
R = 6371  # 地球の半径, km
x1 = [384400+3000, 0, 0, 0.1, 1.022+1.02-2.923, 0]
# x1 = [R + 36000, 0, 0, 0, 3.0668882673431845 / 1.114, 3.0668882673431845]
y1 = [384400, 0, 0, 0, 1.022, 0]

create_orbit_animation(x1, y1, start_fraction=0.0, end_fraction=0.175, step=3000, highlight_latest=True, speed_factor=1)
