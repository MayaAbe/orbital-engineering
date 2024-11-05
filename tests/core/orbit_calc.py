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

def hohmann_transfer_dv(r1, r2, mu=GM):
    """
    Calculate the delta-v values for a Hohmann transfer between two circular orbits.

    Parameters:
    r1: float
        Radius of the initial orbit (in km).
    r2: float
        Radius of the final orbit (in km).
    mu: float, optional
        Gravitational parameter of the central body (in km^3/s^2). Default is Earth's gravitational parameter.

    Returns:
    dv1: float
        Delta-v for the first burn (in km/s).
    dv2: float
        Delta-v for the second burn (in km/s).
    dv_total: float
        Total delta-v for the Hohmann transfer (in km/s).
    """
    # Calculate the velocities in the initial and target orbits
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)

    # Semi-major axis of the transfer ellipse
    a_transfer = (r1 + r2) / 2

    # Velocities at periapsis and apoapsis of the transfer orbit
    v_transfer1 = math.sqrt(mu * 2 * (1 / r1 - 1 / a_transfer))
    v_transfer2 = math.sqrt(abs(mu * 2 * (abs(1 / r2) - abs(1 / a_transfer))))

    # Delta-v for the two burns
    dv1 = abs(v_transfer1 - v1)
    dv2 = abs(v2 - v_transfer2)

    # Total delta-v
    dv_total = dv1 + dv2

    return dv1, dv2, dv_total


# Example usage
r1 = R+200  # Radius of initial orbit in km
r2 = 384400  # 384400  # Radius of final orbit in km

dv1, dv2, dv_total = hohmann_transfer_dv(r1, r2)
print(f"Delta-v1: {dv1:.3f} km/s")
print(f"Delta-v2: {dv2:.3f} km/s")
print(f"Total Delta-v: {dv_total:.3f} km/s")


def calculate_orbit_axes(v, r, mu=GM):
    """
    与えられた速度、重力定数、半径から楕円軌道の長半径と短半径を計算します。

    Parameters:
    v: float
        任意の点での軌道速度（km/s 単位）。
    r: float
        与えられた速度における中央天体から軌道上の物体までの半径（km 単位）。
    mu: float, optional
        中央天体の重力定数（km^3/s^2 単位）。デフォルトは地球の重力定数。

    Returns:
    a: float
        軌道の長半径（km 単位）。
    b: float
        軌道の短半径（km 単位）。
    """
    # 特定軌道エネルギー
    epsilon = v**2 / 2 - mu / r
    print(f"epsilon: {epsilon}")

    # 長半径
    a = -mu / (2 * epsilon)

    # 単位質量あたりの角運動量
    h = r * v

    # 短半径
    if epsilon < 0:
        b = h / math.sqrt(mu) * math.sqrt(abs(a))
    else:
        b = None

    return a, b


# 使用例
vc = v_circular([R+200, 0, 0])
print(f"vc: {vc:.3f} km/s")
v = vc   # 速度（km/s 単位、例: 低軌道速度）
r = R + 200  # 地球の中心からの半径（km 単位、例: 地表から高度300 km）

a, b = calculate_orbit_axes(v, GM, r)
print(f"長半径: {a:.3f} km")
print(f"短半径: {b} km")


if __name__ == '__main__':
    print(hohmann_pos(R + 200, 35786))
