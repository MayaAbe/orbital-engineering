import time
import numpy as np
import matplotlib.pyplot as plt
import core.two_body as tb
import core.orbit_calc as oc
import csv
# 長さの単位はkm, 時間の単位はs

R = 6378.137  # Earth radius
x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]  # 3000km

# 与える初期値は(1)x1, (2)r_aim
# 変化させるのは(1)dv1のx成分, (2)dv1のy成分, (3)月の初期位置
# これらの変数を変化させて目標軌道に到達するまでの "dvの総和が最小となるような初期値" を探す
def grid_search(
    x1, r_aim,  # 初期ベクトルと目標半径(スカラー)
    dv1_x=(-5.0, 5.0),  # dv1のx成分の探索範囲
    dv1_y=(-3.5, -2.5),  # dv1のy成分の探索範囲
    increments=(0.01, 0.01)  # 探索の刻み幅(デフォルトは0.1, 0.01, 1度)
):
    min_dv = float('inf')
    best_params = None
    best_initials = None

    total_iterations = (
        ((dv1_x[1] - dv1_x[0]) / increments[0] + 1) *
        ((dv1_y[1] - dv1_y[0]) / increments[1] + 1)
    )

    current_iteration = 0
    y1 = [384400, 0, 0, 0, 1.022, 0]

    dv_values = []
    time_cost_values = []
    dv1_x_values = []
    dv1_y_values = []

    for dv1_x_value in np.arange(dv1_x[0], dv1_x[1] + increments[0], increments[0]):
        for dv1_y_value in np.arange(dv1_y[0], dv1_y[1] + increments[1], increments[1]):
            dv1 = [dv1_x_value, dv1_y_value, 0]

            dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost = tb.hohman_orbit3(x1, y1, r_aim, dv1)
            abs_dv1 = np.linalg.norm(dv1_ans)
            if dv2_ans[0] is not None:
                abs_dv2 = np.linalg.norm(dv2_ans)
            else:
                abs_dv2 = np.inf
            dv = abs_dv1 + abs_dv2

            dv_values.append(dv)
            time_cost_values.append(time_cost)
            dv1_x_values.append(dv1_x_value)
            dv1_y_values.append(dv1_y_value)

            if np.linalg.norm(sol_com[-1][0:3]) >= r_aim:
                if dv < min_dv:
                    min_dv = dv
                    best_params = [dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost]
                    best_initials = [dv1_x_value, dv1_y_value, 0]

            current_iteration += 1
            print(f'Iteration {current_iteration}/{total_iterations}')
            print(f'Current dv1: {dv1_ans} dv2: {dv2_ans}')
            print(f'Current dv: {dv}')

    return best_params, best_initials, dv_values, time_cost_values, dv1_x_values, dv1_y_values

def save_to_csv(filename, dv1_x_values, dv1_y_values, dv_values, time_cost_values):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["dv1_x", "dv1_y", "dv", "time_cost"])
        for dv1_x, dv1_y, dv, time_cost in zip(dv1_x_values, dv1_y_values, dv_values, time_cost_values):
            writer.writerow([dv1_x, dv1_y, dv, time_cost])

def main():
    param = [
        (0.80, 0.90),   # dv1のx成分の探索範囲
        (-2.7, -2.65),  # dv1のy成分の探索範囲
        (0.0005, 0.0005)
    ]

    # グリッドサーチの実行と時間の計測
    start_time = time.time()
    best_params, best_initials, dv_values, time_cost_values, dv1_x_values, dv1_y_values = grid_search(
        x1, R + 35786, param[0], param[1], param[2]
    )
    end_time = time.time()
    print(f'Time elapsed: {end_time - start_time} seconds')

    # 出力結果のCSVへの保存
    save_to_csv("grid_search_results_detail_x080_090_ym2.7_m2.65.csv", dv1_x_values, dv1_y_values, dv_values, time_cost_values)

    if best_params is not None:
        print(f'Best parameters: {best_params}')
        dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost = best_params
        print(f'best_dv1_initials: {best_initials}')
        print(f'best_dv2: {dv2_ans}')
        dv = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)
        print(f'Best dv: {dv}')
        # 描画
        plt.plot(sol1[:, 0], sol1[:, 1], 'b', label='before dv1')
        plt.plot(sol_com[:, 0], sol_com[:, 1], 'k', label='trajectory')
        plt.plot(sol2[:, 0], sol2[:, 1], 'r--', label='target orbit')
        plt.grid()  # 格子をつける
        plt.gca().set_aspect('equal')  # グラフのアスペクト比を揃える
        plt.legend()
        plt.show()

        energy = np.array([oc.energy(sol_com[i]) for i in range(len(sol_com))])
        plt.figure()
        plt.plot(energy)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Variation')
        plt.grid()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(dv1_x_values, dv1_y_values, dv_values, c=time_cost_values, cmap='viridis', s=1)
        ax.set_xlabel('dv1_x_value')
        ax.set_ylabel('dv1_y_value')
        ax.set_zlabel('dv')
        plt.colorbar(scatter, label='time_cost')
        plt.title('3D plot of dv vs dv1_x_value and dv1_y_value with time_cost')
        plt.show()

    else:
        print("No valid parameters found.")

if __name__ == "__main__":
    main()
