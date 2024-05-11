# Import Python Modules
import numpy as np  # 数値計算ライブラリ
#from scipy.integrate import odeint  # 常微分方程式を解くライブラリ
#import matplotlib.pyplot as plt  # 描画ライブラリ
import orbit_calc as oc  # 自作ライブラリ


def MultiBody(x, y, n):
    T_E = oc.T_circular(x)  # 地球周回衛星の軌道周期
    T_M = oc.T_circular(y)  # 月の軌道周期

    
    t0 = np.linspace(0, Tx, 1000)



def funcTri(x, t):
    GM = 398600.4354360959  # 地球の重力定数, km3/s2
    GMm = 4904.058  # 月の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    r_m = np.linalg.norm(x[6:9])
    z = [x[0]-x[6], x[1]-x[7], x[2]-x[8]]
    r_z = np.linalg.norm(z)
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3) - (GMm*z[0]/(r_z**3) + GMm*x[6]/(r_m**3)),
            -GM*x[1]/(r**3) - (GMm*z[1]/(r_z**3) + GMm*x[7]/(r_m**3)),
            -GM*x[2]/(r**3) - (GMm*z[2]/(r_z**3) + GMm*x[8]/(r_m**3))]
    return dxdt