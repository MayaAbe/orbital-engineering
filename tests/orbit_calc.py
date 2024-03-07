import math
import numpy as np

# Constants
GM = 398600.4354360959 # Earth's gravitational constant
R = 6371000 # Earth radius
T = 86164.1 # Sidereal day
w = 2 * math.pi / T # Angular velocity
h = R + 35800 * 1000 # Altitude
v = math.sqrt(GM / h) # Orbital velocity


# Function
# 位置を引数ととった円軌道の速度
def v_circular(x):
    r = np.linalg.norm(x[0:3])
    return math.sqrt(GM / r)

# 半径を引数ととった円軌道の軌道周期
def T_circular(x):
    r = np.linalg.norm(x[0:3])
    v = v_circular(x[0:3])
    return 2 * math.pi * r / v

# 2インパルスホーマン移行を計算する関数
def hohmann(x1, x2):
    r1=np.linalg.norm(x1[0:3])
    r2=np.linalg.norm(x2[0:3])
    a = (r1 + r2) / 2
    v1 = np.sqrt(GM / r1)
    v2 = np.sqrt(GM / r2)
    v = np.sqrt(GM / a)
    dv1 = v - v1
    dv2 = v2 - v
    return dv1, dv2