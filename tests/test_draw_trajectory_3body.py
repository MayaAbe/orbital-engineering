import core.two_body as tb
import numpy as np
import matplotlib.pyplot as plt

# 初期条件（例）
x1 = [384400 + 3000, 0, 0, 0, 1.022+ 1.02, 0]  # 3000 km
y1 = [384400, 0, 0, 0, 1.022, 0]

# システムの解を求める
solx, soly = tb.MoonEarthSat(x1, y1, 1, 100)

# プロットの作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 解のプロット
ax.plot(solx[:, 0], solx[:, 1], solx[:, 2], 'b', label='Satellite trajectory')
ax.plot(soly[:, 0], soly[:, 1], soly[:, 2], 'r', label='moon trajectory')

# 軸ラベルを設定（単位付き、指数表記をまとめる）
scale_factor = 1e5  # ここで指数の基数を設定（例：1e3 = 10^3）
ax.set_xlabel(f'X (km) x 10^5')
ax.set_ylabel(f'Y (km) x 10^5')
ax.set_zlabel(f'Z (km) x 10^5')

# 目盛りラベルを通常の数値にする
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/scale_factor:.1f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/scale_factor:.1f}'))
ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/scale_factor:.1f}'))

plt.grid()  # グリッドを有効にする
plt.legend()
plt.gca().set_aspect('auto')  # 読みやすさのためにアスペクト比を調整

plt.show()
