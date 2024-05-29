import two_body as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

R = 6371  # 地球の半径, km
x1 = [R + 36000, 0, 0, 0, 3.0668882673431845 / 1.114, 3.0668882673431845]
y1 = [384400, 0, 0, 0, 1.022, 0]

solx, soly = tb.MoonEarthSat(x1, y1, 60, 10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 軌道の範囲を設定
ax.set_xlim([min(solx[:, 0]), max(solx[:, 0])])
ax.set_ylim([min(solx[:, 1]), max(solx[:, 1])])
ax.set_zlim([min(solx[:, 2]), max(solx[:, 2])])

# 軌道の描画
line_solx, = ax.plot([], [], [], 'b')
line_soly, = ax.plot([], [], [], 'r')

def init():
    line_solx.set_data([], [])
    line_solx.set_3d_properties([])
    line_soly.set_data([], [])
    line_soly.set_3d_properties([])
    return line_solx, line_soly

def update(frame):
    line_solx.set_data(solx[:frame, 0], solx[:frame, 1])
    line_solx.set_3d_properties(solx[:frame, 2])
    line_soly.set_data(soly[:frame, 0], soly[:frame, 1])
    line_soly.set_3d_properties(soly[:frame, 2])
    return line_solx, line_soly

# 100倍速くするためにintervalを小さくする
ani = FuncAnimation(fig, update, frames=len(solx), init_func=init, blit=True, interval=1)

plt.grid()  # 格子をつける
plt.show()
