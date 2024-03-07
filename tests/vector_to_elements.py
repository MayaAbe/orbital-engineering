import numpy as np
GM = 398600.4354360959 # 重力定数(地球)


def v2e(t=0, r=0, v=0):
    # 位置と速度を指定
    t = 0
    r = np.array([0, -3299.603, -6599.209])
    v = np.array([9.645803, 0, 0])

    # 軌道半径と軌道エネルギーから班長軸を求める
    r_norm = np.linalg.norm(r)
    ep = np.dot(v, v) / 2 - GM / r_norm
    a = -GM / (2 * ep)

    # 角運動量ベクトルを求める
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    # ラプラスベクトルを求める
    P = -(GM/r_norm)*r - np.cross(h, v)
    P_norm = np.linalg.norm(P)
    P_hat = P / P_norm

    # 離心率を求める
    e = P_norm / GM

    # 離心近点離角を求める
    e_cosE = 1 - r_norm / a
    e_sinE = np.dot(r, v) / np.sqrt(GM * a)
    E = np.arctan2(e_sinE, e_cosE)

    # 軌道傾斜角を求める
    W_hat = h / h_norm
    i = np.arccos(np.dot( W_hat,([0, 0, 1]) ))
    i = np.degrees(i)

    # 昇交点赤経を求める
    Omega = np.arctan2(np.dot(W_hat,([1,0,0])), np.dot(-W_hat,([0,1,0])))
    Omega = np.degrees(Omega)

    # 近点引数を求める
    omega = np.arctan2(np.dot(P_hat, W_hat), np.dot(P_hat, ([1,0,0])))
    omega = np.degrees(omega)

    return t, e, E, i, Omega, omega

# 結果を表示
if __name__ == '__main__':
    [t, e, E, i, Omega, omega] = v2e()
    print("近心点通過時刻：{:.3f}".format(t))
    print("離心率：{:.3f}".format(e))
    print("離心近点離角：{:.3f}".format(E))
    print("軌道傾斜角：{:.3f}".format(i))
    print("昇交点赤経：{:.3f}".format(Omega))
    print("近点引数：{:.3f}".format(omega))