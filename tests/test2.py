import two_body as tb
import numpy as np
import matplotlib.pyplot as plt

R = 6371  # 地球の半径, km
x1 = [R+36000, 0, 0, 0, 3.0668882673431845/1.414, 3.0668882673431845/1.414]
y1 = [384400, 0, 0, 0, 1.022, 0]

solx, soly = tb.MoonEarthSat(x1, y1, 10000, 100000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solx[:, 0], solx[:, 1], solx[:, 2], 'b')
ax.plot(soly[:, 0], soly[:, 1], soly[:, 2], 'r')
plt.grid()  # 格子をつける
plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
plt.show()