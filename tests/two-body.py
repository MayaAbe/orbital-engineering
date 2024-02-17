# Import Python Modules
import numpy as np # 数値計算ライブラリ
from scipy.integrate import odeint # 常微分方程式を解くライブラリ
import matplotlib.pyplot as plt # 描画ライブラリ

# 二体問題の運動方程式
def func(x, t):
    GM = 398600.4354360959 # 地球の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3),
            -GM*x[1]/(r**3),
            -GM*x[2]/(r**3)]
    return dxdt

# 微分方程式の初期条件
x0 = [6400+36000, 0, 0, 0, 4, 0] # 位置(x,y,z)＋速度(vx,vy,vz)
t  = np.linspace(0, 86400, 1000) # 1日分 軌道伝播

# 微分方程式の初期条件
x1 = [6400, 0, 0, 0, 7.9, 0] # 位置(x,y,z)＋速度(vx,vy,vz)
t1  = np.linspace(0, 86400, 1000) # 1日分 軌道伝播

# 微分方程式の数値計算
sol = odeint(func, x0, t)

sol1 = odeint(func, x1, t1)

# 描画
plt.plot(sol[:, 0],sol[:, 1], 'b')
plt.plot(sol1[:, 0],sol1[:, 1], 'r')
plt.grid() # 格子をつける
plt.gca().set_aspect('equal') # グラフのアスペクト比を揃える
plt.show()
