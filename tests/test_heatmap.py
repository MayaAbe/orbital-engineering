import time
import numpy as np
import matplotlib.pyplot as plt
import core.two_body as tb
import core.orbit_calc as oc

# 初期ベクトルと定数の設定
R = 6378.137  # Earth radius
x1 = [R + 200, 0, 0, 0, 7.784261686425335, 0]

def grid_search(x1, r_aim, dv1_x=(-0.1, 0.1), dv1_y=(2.2, 2.7), increments=(0.01, 0.01)):
    min_dv = float('inf')
    best_params = None
    best_initials = None

    total_iterations = (
        ((dv1_x[1] - dv1_x[0]) / increments[0] + 1) *
        ((dv1_y[1] - dv1_y[0]) / increments[1] + 1)
    )

    current_iteration = 0

    # dv 値を格納するリストを初期化
    dv_values = []

    for dv1_x_value in np.arange(dv1_x[0], dv1_x[1] + increments[0], increments[0]):
        row = []
        for dv1_y_value in np.arange(dv1_y[0], dv1_y[1] + increments[1], increments[1]):
            dv1 = [dv1_x_value, dv1_y_value, 0]

            # dv1を計算
            dv1_ans, dv2_ans, sol_com, sol1, sol2 = tb.hohman_orbit2(x1, r_aim, dv1)
            abs_dv1 = np.linalg.norm(dv1_ans)
            abs_dv2 = np.linalg.norm(dv2_ans)
            dv = abs_dv1 + abs_dv2

            # sol_comの末項で半径がr_aim kmを超えるかどうかを確認
            if np.linalg.norm(sol_com[-1][0:3]) >= r_aim:
                if dv < min_dv:
                    min_dv = dv
                    best_params = [dv1_ans, dv2_ans, sol_com, sol1, sol2]
                    best_initials = [dv1_x_value, dv1_y_value, 0]

            row.append(dv)

            current_iteration += 1
            print(f'Iteration {current_iteration}/{total_iterations}')
            print(f'Current dv1: {dv1_ans} dv2: {dv2_ans}')
            print(f'Current dv: {dv}')

        dv_values.append(row)

    return best_params, best_initials, dv_values

# 実行
start_time = time.time()
best_params, best_initials, dv_values = grid_search(x1, R + 35786)
end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds')

# dv値のヒートマップを描画
dv_values = np.array(dv_values)
dv_values = dv_values.T  # 転置して形状を整える
x = np.arange(-0.1, 0.1 + 0.01, 0.01)
y = np.arange(2.2, 2.7 + 0.01, 0.01)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, dv_values, levels=100, cmap='viridis')
plt.colorbar(label='dv')
plt.xlabel('dv1_x')
plt.ylabel('dv1_y')
plt.title('Heatmap of dv')
plt.show()

if best_params is not None:
    print(f'Best parameters: {best_params}')
    dv1_ans, dv2_ans, sol_com, sol1, sol2 = best_params
    print(f'best_dv1_initials: {best_initials}')
    dv = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)
    print(f'Best dv: {dv}')
    # 描画
    plt.plot(sol_com[:, 0], sol_com[:, 1], 'k')
    plt.plot(sol1[:, 0], sol1[:, 1], 'b')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r--')
    plt.grid()  # 格子をつける
    plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
    plt.show()

    # エネルギーの時間変化を描画
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
