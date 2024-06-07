import time
import numpy as np
import matplotlib.pyplot as plt
import two_body as tb
import orbit_calc as oc
# 長さの単位はkm, 時間の単位はs

R = 6378.137 # Earth radius
x1 = [R+200, 0, 0, 0, 7.784261686425335, 0]
# dv1 = [0, 2.5 ,0]


# 与える初期値は(1)x1, (2)r_aim
# 変化させるのは(1)dv1のx成分, (2)dv1のy成分, (3)月の初期位置
# これらの変数を変化させて目標軌道に到達するまでの "dvの総和が最小となるような初期値" を探す
def grid_search(
    x1, r_aim,  # 初期ベクトルと目標半径(スカラー)
    dv1_x=(0, 1),  # dv1のx成分の探索範囲
    dv1_y=(2.8, 3.3),  # dv1のy成分の探索範囲
    y1_theta=(0, 2*np.pi),  # 月の初期位置の探索範囲
    increments=(0.1, 0.1, 2*np.pi/180)  # 探索の刻み幅(デフォルトは0.1, 0.01, 1度)
):
    min_dv = float('inf')
    best_params = None
    best_initials = None

    total_iterations = (
        ((dv1_x[1] - dv1_x[0]) / increments[0]+1) *
        ((dv1_y[1] - dv1_y[0]) / increments[1]+1) *
        ((y1_theta[1] - y1_theta[0]) / increments[2]+1)
    )

    current_iteration = 0

    for dv1_x_value in np.arange(dv1_x[0], dv1_x[1]+increments[0], increments[0]):
        for dv1_y_value in np.arange(dv1_y[0], dv1_y[1]+increments[1], increments[1]):
            for y1_theta_value in np.arange(y1_theta[0], y1_theta[1]+increments[2], increments[2]):
                y1 = [384400*np.cos(y1_theta_value), 384400*np.sin(y1_theta_value), 0, -1.022 * np.sin(y1_theta_value), 1.022 * np.cos(y1_theta_value), 0]
                dv1 = [dv1_x_value, dv1_y_value, 0]

                dv1_ans, dv2_ans, sol_com, sol1, sol2 = tb.hohman_orbit3(x1, y1, r_aim, dv1)
                print(f'dv1_ans: {dv1_ans}, dv2_ans: {dv2_ans}')
                abs_dv1 = np.linalg.norm(dv1_ans)
                abs_dv2 = np.linalg.norm(dv2_ans)
                dv = abs_dv1 + abs_dv2

                # sol_comの末項で半径がr_aimkmを超えるかどうか
                if np.linalg.norm(sol_com[-1][0:3]) >= (r_aim-10000):
                    if dv < min_dv:
                        min_dv = dv
                        best_params = [dv1_ans, dv2_ans, sol_com, sol1, sol2, y1_theta_value]
                        best_initials = [dv1_x_value, dv1_y_value, 0]
                #else :
                    #np.linalg.norm(sol_com[-1][0:3])

                current_iteration += 1
                print(f'Iteration {current_iteration}/{total_iterations}')
                print(f'Currenr dv1: {dv1_ans} dv2: {dv2_ans}')
                print(f'Current dv: {dv}')

    return best_params, best_initials


start_time = time.time()
best_params, best_initials = grid_search(x1, R + 384400/5)
end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds')

if best_params is not None:
    print(f'Best parameters: {best_params}')
    dv1_ans, dv2_ans, sol_com, sol1, sol2, theta = best_params
    print(f'best_dv1_initials: {best_initials}')
    dv = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)
    print(f'Best dv: {dv}')
    print(f'Best theta: {theta}')
    # 描画
    plt.plot(sol_com[:, 0], sol_com[:, 1], 'k')
    plt.plot(sol1[:, 0], sol1[:, 1], 'b')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r--')
    plt.grid()  # 格子をつける
    plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
    plt.show()

    energy = np.array([oc.energy(sol_com[i]) for i in range(len(sol_com))])
    plt.figure()
    plt.plot(energy)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Variation')
    plt.grid()
    plt.show()
else:
    print("No valid parameters found.")
