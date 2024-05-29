# Import Python Modules
import numpy as np  # 数値計算ライブラリ
from scipy.integrate import odeint  # 常微分方程式を解くライブラリ
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt  # 描画ライブラリ
import orbit_calc as oc  # 自作ライブラリ

r_E = 6371  # 地球の半径, km

# 二体問題の運動方程式
def func(x, t):
    GM = 398600.4354360959  # 地球の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3),
            -GM*x[1]/(r**3),
            -GM*x[2]/(r**3)]
    return dxdt


# 二体問題の運動方程式
def funcMoon(x, t):
    GM = 403493.253  # 月の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3),
            -GM*x[1]/(r**3),
            -GM*x[2]/(r**3)]
    return dxdt

def funcTri(x, t, y_interpolated):
    y = y_interpolated(t)
    GM = 398600.4354360959  # 地球の重力定数, km3/s2
    GMm = 4904.058  # 月の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    r_m = np.linalg.norm(y[0:3])
    z = [x[0]-y[0], x[1]-y[1], x[2]-y[2]]
    r_z = np.linalg.norm(z)
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3) - (GMm*z[0]/(r_z**3) + GMm*y[0]/(r_m**3)),
            -GM*x[1]/(r**3) - (GMm*z[1]/(r_z**3) + GMm*y[1]/(r_m**3)),
            -GM*x[2]/(r**3) - (GMm*z[2]/(r_z**3) + GMm*y[2]/(r_m**3))]
    return dxdt

def MoonEarthSat(x: tuple, y: tuple, n: int, step: int):
    Tx = oc.T_circular(x)  # 地球公転周期
    Ty = oc.T_circular(y)  # 月公転周期

    # m*Ty > n*Tx となる最小のmを探す
    m = 1
    while m*Ty < n*Tx:
        m += 1
    tx = np.linspace(0, n*Tx, int(n*Tx/step))  # 地球公転周期分
    ty = np.linspace(0, m*Ty, int(m*Ty/step))  # 月公転周期分

    soly = odeint(funcMoon, y, ty)
    y_interpolated = interp1d(ty, soly, axis=0, kind='cubic', fill_value="extrapolate")
    solx = odeint(lambda x, t: funcTri(x, t, y_interpolated), x, tx)
    solx = np.asarray(solx)
    soly = np.asarray(soly)
    return solx, soly

def draw_hohman_orbit(x1, x2, tr):
    # 微分方程式の初期条件
    t1  = np.linspace(0, oc.T_circular(x1), 1000)  # 1日分 軌道伝播
    sol1 = odeint(func, x1, t1)

    t2 = np.linspace(0, oc.T_circular(x2), 1000)  # 1日分 軌道伝播
    sol2 = odeint(func, x2, t2)

    ttr = np.linspace(0, oc.T_owbow(x1), 1000)
    soltr = odeint(func, tr, ttr)
    soltr = trim_solution(soltr, np.linalg.norm(x2[0:3]))

    # 描画
    plt.plot(soltr[:, 0], soltr[:, 1], 'k')
    plt.plot(sol1[:, 0], sol1[:, 1], 'b')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r')
    plt.grid()  # 格子をつける
    plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
    plt.show()


def trim_solution(sol, r_aim):
    # Iterate over the solution to find the index where the norm exceeds 'r2'
    print(np.linalg.norm(sol[2][:3]))
    print(1)
    for i, x1 in enumerate(sol):
        if np.linalg.norm(x1[:3]) > r_aim:
            new_sol = sol[:i]
            # 倍率をt(0~1)とし，以下の変数new_pointについて二分探索でr2==np.linalg.norm(new_point[:3])になるようにする
            # new_point = new_sol[-1] + t * (sol[i] - new_sol[-1])
            # このtを求める
            # 二分探索の初期値
            left = 0
            right = 1
            print(type(new_sol))
            print(type(sol))
            print(f"new_sol shape: {new_sol.shape}")
            print(f"new_sol[-1] shape: {new_sol[-1].shape}")
            print(new_sol[-1])
            print(f"sol shape: {sol.shape}")
            print(f"sol[{i}] shape: {sol[i].shape}")
            print(sol[i])
            #print(sol[i] - new_sol[-1])
            # 二分探索
            while right - left > np.finfo(np.float64).eps:
                t = (left + right) / 2
                new_point = new_sol[-1] + t * (sol[i] - new_sol[-1])
                if np.linalg.norm(new_point[:3]) > r_aim:
                    right = t
                else:
                    left = t
            new_point = new_sol[-1] + right * (sol[i] - new_sol[-1])
            print(np.linalg.norm(new_point[:3]))
            # Return the modified solution with the new interpolated point as the last point
            return np.concatenate((new_sol, [new_point]))
    return sol
    # Return the original solution if no elements exce    return sol



def draw_hohman_orbit2(x1, r2, dv1):
    # 与えられた円軌道を軌道伝播
    t1 = np.linspace(0, oc.T_circular(x1), 1000)
    sol1 = odeint(func, x1, t1)
    # 目標軌道を書く
    x2 = [r_E+35786, 0, 0]
    x2 = [r_E+35786, 0, 0, 0, oc.v_circular(x2), 0]
    t2  = np.linspace(0, oc.T_circular(x2), 1000)  # 1日分 軌道伝播
    sol2 = odeint(func, x2, t2)

    tr = x1.copy()
    # print(tr)
    tr[3] += dv1[0]; tr[4] += dv1[1]; tr[5] += dv1[2]
    # print(tr)
    # 遷移軌道の楕円部分dv1影響後
    # 1.双曲線軌道に入ったらとりあえず8000秒書く
    if oc.T_owbow(tr) == np.inf:
        # print("T is infty")
        ttr = np.linspace(0, 8000, 80000)
    # 2.双曲線軌道に入らなければ大体OK
    else:
        ttr = np.linspace(0, oc.T_owbow(tr), int(oc.T_owbow(tr)/10))
        # print("T is "+str(oc.T_owbow(tr)))
    soltr = odeint(func, tr, ttr)
    print(type(soltr))
    # 楕円軌道が目標軌道半径まで到達していたらそれ以降を削除
    soltr = trim_solution(soltr, r2)
    # print(soltr)

    # 遷移後の軌道dv2影響後
    v_x2 = oc.v_circular(soltr[-1])
    # dv2遷移時の速度ベクトルの角度
    theta = np.arctan2(soltr[-1][1], soltr[-1][0])
    # dv2以降の軌道のカルテシアン要素
    tr_x2 = [soltr[-1][0], soltr[-1][1], soltr[-1][2], -v_x2*np.sin(theta), v_x2*np.cos(theta), 0]
    # 軌道伝播時間は円軌道半周期分
    ttr2 = np.linspace(0, oc.T_circular(tr_x2)/2, 1000)
    soltr2 = odeint(func, tr_x2, ttr2)
    # dv1遷移軌道とdev2遷移軌道を連結する
    soltr_combined = np.concatenate((soltr, soltr2))

    # ベクトルの差分をとってからノルムを計算する
    # 遷移軌道末項位置における円軌道の速度ベクトル-楕円軌道の速度ベクトル
    dv2 =(tr_x2[3:6] -soltr[-1][3:6]).tolist()


    # 描画
    plt.plot(soltr_combined[:, 0], soltr_combined[:, 1], 'k')
    plt.plot(sol1[:, 0], sol1[:, 1], 'b')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r--')
    plt.grid()  # 格子をつける
    plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
    plt.show()
    return dv1, dv2, soltr_combined, sol1, sol2


def draw_hohman_orbit3(x1, r2, dv1):
    # 与えられた円軌道を2体問題で軌道伝播
    t1 = np.linspace(0, oc.T_circular(x1), 1000)
    sol1 = odeint(func, x1, t1)
    # 2体問題で目標軌道を書く
    r_x2 = [r2, 0, 0]
    x2 = [r2, 0, 0, 0, oc.v_circular(r_x2), 0]
    t2  = np.linspace(0, oc.T_circular(x2), 1000)  # 1日分 軌道伝播
    sol2 = odeint(func, x2, t2)

    tr = x1.copy()
    # print(tr)
    # 速度増分dv1を適用
    tr[3] += dv1[0]; tr[4] += dv1[1]; tr[5] += dv1[2]
    print(tr)
    # 遷移軌道の楕円部分dv1影響後
    # 1.双曲線軌道に入ったらとりあえず8000秒を0.1秒ステップで書く
    if oc.T_owbow(tr) == np.inf:
        n = 80
        step = 0.1
    # 2.加速された周期が目標周期より小さければ
    else:
        n = 10
        step = 0.1
    # 楕円軌道が目標軌道半径まで到達していたらそれ以降を削除
    print(f"n=:{n}step=:{step}")
    sol, a = MoonEarthSat(tr, x2, n, step)
    print(sol)
    soltr = trim_solution(sol, r2)
    print(soltr)

    # 遷移後の軌道dv2影響後
    v_x2 = oc.v_circular(soltr[-1])
    # dv2遷移時の速度ベクトルの角度
    theta = np.arctan2(soltr[-1][1], soltr[-1][0])
    # dv2以降の軌道のカルテシアン要素
    tr_x2 = [soltr[-1][0], soltr[-1][1], soltr[-1][2], -v_x2*np.sin(theta), v_x2*np.cos(theta), 0]
    #t2_adjusted = np.linspace(0, oc.T_circular(tr_x2), len(soltr))
    # 軌道伝播時間は円軌道半周期分
    soltr2, b = MoonEarthSat(tr_x2, x2, 1, 10)
    #print(soltr2)
    # dv1遷移軌道とdev2遷移軌道を連結する
    print(f"soltr shape: {soltr.shape}")
    print(f"soltr2 shape: {soltr2.shape}")
    soltr_combined = np.concatenate((soltr, soltr2))

    # ベクトルの差分をとってからノルムを計算する
    # 遷移軌道末項位置における円軌道の速度ベクトル-楕円軌道の速度ベクトル
    dv2 = (tr_x2[3:6] - soltr[-1][3:6]).tolist()

    plt.plot(soltr_combined[:, 0], soltr_combined[:, 1], 'k')
    plt.plot(sol1[:, 0], sol1[:, 1], 'b')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r--')
    plt.grid()  # 格子をつける
    plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
    plt.show()
    return dv1, dv2, soltr_combined, sol1, sol2


if __name__ == '__main__':
    # 地球の輪郭
    # x_E = [r_E, 0, 0, 0, 7.9, 0] # 位置(x,y,z)＋速度(vx,vy,vz)
    # t_E  = np.linspace(0, 5100, 100) # 1日分 軌道伝播
    # solE = odeint(func, x_E, t_E)

    # 微分方程式の初期条件
    x0 = [r_E+1000, 0, 0]  # 位置(x,y,z)
    x0 = [r_E+1000, 0, 0, 0, oc.v_circular(x0), 0]  # 位置(x,y,z)＋速度(vx,vy,vz)
    t0 = np.linspace(0, oc.T_circular(x0), 10000000)  # 1日分 軌道伝播
    sol0 = odeint(func, x0, t0)
    print(sol0)
    # 描画
    # plt.plot(solE[:, 0],solE[:, 1],'k')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol0[:, 0], sol0[:, 1], sol0[:, 2], 'b')
    plt.grid()  # 格子をつける
    plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
    plt.show()
