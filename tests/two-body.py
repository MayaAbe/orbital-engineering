# Import Python Modules
import numpy as np # 数値計算ライブラリ
from scipy.integrate import odeint # 常微分方程式を解くライブラリ
import matplotlib.pyplot as plt # 描画ライブラリ
import orbit_calc as oc # 自作ライブラリ

r_E = 6371 # 地球の半径, km

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

# 地球の輪郭
x_E = [r_E, 0, 0, 0, 7.9, 0] # 位置(x,y,z)＋速度(vx,vy,vz)
t_E  = np.linspace(0, 5100, 100) # 1日分 軌道伝播
solE = odeint(func, x_E, t_E)

# 微分方程式の初期条件
x0 = [r_E+1000, 0, 0, 0, 4, 0] # 位置(x,y,z)＋速度(vx,vy,vz)
x0 = [r_E+1000, 0, 0, 0, oc.v_circular(x0), 0] # 位置(x,y,z)＋速度(vx,vy,vz)
t0  = np.linspace(0, oc.T_circular(x0), 1000) # 1日分 軌道伝播

# 微分方程式の初期条件
x1= [r_E+400, 0, 0, 0, 4, 0] # 位置(x,y,z)＋速度(vx,vy,vz)
x1 = [r_E+400, 0, 0, 0, oc.v_circular(x1), 0] # 位置(x,y,z)＋速度(vx,vy,vz)
t1  = np.linspace(0, oc.T_circular(x1), 1000) # 1日分 軌道伝播

# 微分方程式の数値計算
sol0 = odeint(func, x0, t0)
sol1 = odeint(func, x1, t1)

# 描画
plt.plot(solE[:, 0],solE[:, 1],'k')
plt.plot(sol0[:, 0],sol0[:, 1], 'b')
plt.plot(sol1[:, 0],sol1[:, 1], 'r')
plt.grid() # 格子をつける
plt.gca().set_aspect('equal') # グラフのアスペクト比を揃える
plt.show()
