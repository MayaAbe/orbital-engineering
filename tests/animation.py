import two_body as tb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation

R = 6371  # 地球の半径, km
x1 = [R + 36000, 0, 0, 0, 3.0668882673431845 / 1.114, 3.0668882673431845]
# x1 = [384400+3000, 0, 0, 0, 1.022+1.02, 0]
y1 = [384400, 0, 0, 0, 1.022, 0]

solx, soly = tb.MoonEarthSat(x1, y1, 60, 10)

# 表示するフレームの範囲を設定（後半部分）
start_fraction = 0.4  # 後半の50%を表示する場合
start_index = int(len(solx) * start_fraction)

solx = solx[start_index:]
soly = soly[start_index:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 軌道の範囲を設定
ax.set_xlim([min(solx[:, 0]), max(solx[:, 0])])
ax.set_ylim([min(solx[:, 1]), max(solx[:, 1])])
ax.set_zlim([min(solx[:, 2]), max(solx[:, 2])])

frames = []
step = 1000  # フレームの間隔を大きくして、メモリ使用量を減らす
for i in range(0, len(solx), step):  # 100倍速くするためにフレーム数を減らす
    line_solx, = ax.plot(solx[:i, 0], solx[:i, 1], solx[:i, 2], 'b')
    line_soly, = ax.plot(soly[:i, 0], soly[:i, 1], soly[:i, 2], 'r')
    frames.append([line_solx, line_soly])

ani = ArtistAnimation(fig, frames, interval=10, blit=True)

plt.grid()  # 格子をつける
plt.show()