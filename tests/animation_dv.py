import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import core.two_body as tb
import core.orbit_calc as oc

def create_orbit_animation(solx_list, soly, solz, start_fraction=0.0, end_fraction=1.0, step=1, interval=10,
                           highlight_latest=True, speed_factor=1.0, save=False, filename='orbit_animation.mp4'):
    """
    複数の solx を同時にアニメーション表示する関数。

    Parameters:
    - solx_list: list of numpy arrays
        複数の solx データのリスト
    - soly: numpy array
        soly データ
    - solz: numpy array
        solz データ（アニメーションしない）
    - その他のパラメータは従来通り
    """
    num_solx = len(solx_list)
    len_soly = len(soly)

    # 各 solx の長さを取得し、最大長を計算
    len_solx_list = [len(solx) for solx in solx_list]
    max_length = max(len_solx_list)
    total_frames = (max_length + 1) // step  # 総フレーム数

    fig, ax = plt.subplots()

    # 軌道の範囲を設定
    all_x = np.concatenate([solx[:, 0] for solx in solx_list] + [soly[:, 0], solz[:, 0]])
    all_y = np.concatenate([solx[:, 1] for solx in solx_list] + [soly[:, 1], solz[:, 1]])

    # マージンの割合を設定（10%）
    margin_ratio = 0.1

    # x軸の範囲とマージンを計算
    x_min, x_max = np.min(all_x), np.max(all_x)
    x_range = x_max - x_min
    x_margin = x_range * margin_ratio
    ax.set_xlim([x_min - x_margin, x_max + x_margin])

    # y軸の範囲とマージンを計算
    y_min, y_max = np.min(all_y), np.max(all_y)
    y_range = y_max - y_min
    y_margin = y_range * margin_ratio
    ax.set_ylim([y_min - y_margin, y_max + y_margin])

    ax.set_aspect('equal')

    # 原点に大きな点をプロット
    origin_scatter = ax.scatter(0, 0, color='k', s=100)

    # solzを最初から全体をプロット
    line_solz_full, = ax.plot(solz[:, 0], solz[:, 1], 'g-', label='target_orbit')

    # 各 solx のラインオブジェクトとマーカーを初期化
    lines_solx = []
    currents_solx = []
    colors = ['b', 'c', 'm', 'y', 'k']  # 複数の色を用意
    for idx, solx in enumerate(solx_list):
        color = colors[idx % len(colors)]
        line_solx, = ax.plot([], [], color + '-', linewidth=1, label=f'solx_{idx}')
        current_solx, = ax.plot([], [], color + 'o', markersize=8)
        lines_solx.append(line_solx)
        currents_solx.append(current_solx)

    # soly のラインオブジェクトとマーカーを初期化
    line_soly, = ax.plot([], [], 'r-', linewidth=1, label='moon_orbit')
    current_soly, = ax.plot([], [], 'ro', markersize=8)

    # 凡例を追加
    ax.legend()

    def init():
        for line_solx, current_solx in zip(lines_solx, currents_solx):
            line_solx.set_data([], [])
            current_solx.set_data([], [])
        line_soly.set_data([], [])
        current_soly.set_data([], [])
        return lines_solx + currents_solx + [line_soly, current_soly, line_solz_full, origin_scatter]

    def animate(frame):
        # フレームに応じてインデックスを計算
        all_solx_completed = True  # すべての solx が描画終了したかを判定

        # 各 solx のデータを更新
        for idx, (solx, line_solx, current_solx) in enumerate(zip(solx_list, lines_solx, currents_solx)):
            index_solx = frame * step
            if index_solx < len(solx):
                x_data_solx = solx[:index_solx+1, 0]
                y_data_solx = solx[:index_solx+1, 1]
                line_solx.set_data(x_data_solx, y_data_solx)
                current_solx.set_data([solx[index_solx, 0]], [solx[index_solx, 1]])
                all_solx_completed = False  # まだ描画中の solx がある
            else:
                # 描画終了した solx はそのまま表示
                line_solx.set_data(solx[:, 0], solx[:, 1])
                current_solx.set_data([], [])  # マーカーを非表示

        # soly のデータを更新（常にループ）
        index_soly = (frame * step) % len_soly
        x_data_soly = soly[:index_soly+1, 0]
        y_data_soly = soly[:index_soly+1, 1]
        line_soly.set_data(x_data_soly, y_data_soly)
        current_soly.set_data([soly[index_soly, 0]], [soly[index_soly, 1]])

        # すべての solx の描画が終了した場合、アニメーションをリセット
        if all_solx_completed:
            return init()

        artists = lines_solx + currents_solx + [line_soly, current_soly, line_solz_full, origin_scatter]
        return artists

    ani = FuncAnimation(fig, animate, init_func=init, frames=total_frames * 10, interval=interval * speed_factor,
                        blit=True, repeat=False)

    if save:
        if filename.endswith('.mp4'):
            Writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(filename, writer=Writer)
        elif filename.endswith('.gif'):
            ani.save(filename, writer='pillow')
        else:
            raise ValueError("Unsupported file format. Please use .mp4 or .gif")

    plt.grid()
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('Orbit Animation')
    plt.show()

# 使用例
# 長さの単位はkm, 時間の単位はs

R = 6378.137  # 地球半径
x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]  # 3000km
y1 = [384400, 0, 0, 0, 1.022, 0]


# dv1 のリストを作成
# n=1
"""
dv1_list = [
    [0.8699999999998749, -2.6700000000000177, 0],  # 元の dv1
    [0.871, -2.6700000000000177, 0],
    [0.868, -2.6700000000000177, 0],
]"""

# n=7

dv1_list = [
    [0.8899999999998744,-2.660000000000018, 0],  # 元の dv1
    [0.8899999999998744-0.01,-2.660000000000018+0.01, 0],
    [0.8899999999998744+0.01,-2.660000000000018-0.01, 0],
]


# solx_list を作成
solx_list = []
for dv1 in dv1_list:
    dv1_result, dv2_result, solx, soly, solz, f = tb.hohman_orbit3(x1, y1, R + 35786, dv1, True)
    solx_list.append(solx)

# アニメーションを作成
create_orbit_animation(solx_list, soly, solz, start_fraction=0.0, end_fraction=1.0,
                       step=10, highlight_latest=True, speed_factor=0.05, save= False)# True, filename='orbit_animation_n1.mp4')
