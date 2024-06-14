import math
import numpy as np

# Constants
GM = 398600.4354360959 # Earth's gravitational constant
R = 6378.137 # Earth radius
T = 86164.1 # Sidereal day
w = 2 * math.pi / T # Angular velocity
h = R + 35800 * 1000 # Altitude
v = math.sqrt(GM / h) # Orbital velocity


# Function
# 任意の位置で円軌道が成立する時の速さ
def v_circular(x):
    r = np.linalg.norm(x[0:3])
    return math.sqrt(GM / r)

# 任意の位置における円軌道が成立するときの軌道周期
def T_circular(x):
    r = np.linalg.norm(x[0:3])
    v = v_circular(x[0:3])
    return 2 * math.pi * r / v


def T_owbow(x):
    # 軌道半径と軌道エネルギーから半長軸を求める
    r_norm = np.linalg.norm(x[0:3])
    ep = np.dot(x[3:6], x[3:6]) / 2 - GM / r_norm
    a = -GM / (2 * ep)
    # 半長軸を求めたら軌道周期T=2π√a3/GMで周期を求める
    if a <= 0:
        return np.inf
    else:
        return 2 * math.pi * np.sqrt(a**3/GM)

# 平面2体問題2インパルスホーマン移行に限ったときのみ使用可能
def T_owbow2(r1, r2):
    r = (r1 + r2) / 2 # 楕円の半長軸を求める
    return 2 * math.pi * np.sqrt(r**3 / GM)

def energy(x):
    r = np.linalg.norm(x[0:3])
    v = np.linalg.norm(x[3:6])
    return v**2 / 2 - GM / r

# 2インパルスホーマン移行において，遷移開始高度と軌道半径の差分をインプットとしたとき，終端位置の座標を返す関数
def hohmann_pos(r1, dr):
    # r1 = np.linalg.norm(x1[0:3])  # 遷移開始位置の軌道半径
    r2 = r1 + dr  # 遷移終了位置の軌道半径
    a = (r1 + r2) / 2  # 遷移軌道の半長軸
    v1 = np.sqrt(GM / r1)  # 遷移開始位置の速さ
    v2 = np.sqrt(GM / r2)  # 遷移終了位置の速さ
    vp = np.sqrt(2*GM*(1/r1 - 1/(r1+r2)))
    va = np.sqrt(2*GM*(1/r2 - 1/(r1+r2)))
    dv1 = vp - v1  # 遷移開始位置での速度変化
    dv2 = v2 - va  # 遷移終了位置での速度変化
    x1 = [r1, 0, 0, 0, v1, 0]  # 遷移開始位置の座標
    x2 = [-r2, 0, 0, 0, -v2, 0]  # 遷移終了位置の座標
    tr = [r1, 0, 0, 0, vp, 0]  # 遷移軌道出発時のカルテシアン要素
    return x1, x2, tr


if __name__ == '__main__':
    print(hohmann_pos(R + 200, 35586))
