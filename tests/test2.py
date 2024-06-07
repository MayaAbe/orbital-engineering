import two_body as tb
import numpy as np
import matplotlib.pyplot as plt

# x1 = [R+200, 0, 0, 0, 7.784261686425335, 0]
# x1 = [384400+5000, 0, 0, 0, 1.022+0.85/1.414, 0.85+0.2]  # naname
x1 = [384400+3000, 0, 0, 0, 1.022+1.02/1.414, 1.02/1.414]  # 3000km
y1 = [384400, 0, 0, 0, 1.022, 0]


# 描画
# plt.plot(solE[:, 0],solE[:, 1],'k')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

solx, soly = tb.MoonEarthSat(x1, y1, 1, 100)
# fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solx[:, 0], solx[:, 1], solx[:, 2], 'b')
ax.plot(soly[:, 0], soly[:, 1], soly[:, 2], 'r')
plt.grid()  # 格子をつける
plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
plt.show()