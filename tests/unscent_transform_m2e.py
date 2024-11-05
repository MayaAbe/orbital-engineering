import pandas as pd
import core.two_body as tb
import stochastic.unscented_transform as ut
import numpy as np

def dv_minmax(filename, n, ascending=True):
    """
    指定されたCSVファイルから、dv列を基準に並び替えたデータのn番目の情報を取得する関数。

    引数:
        filename: CSVファイルのパス
        n: n番目の値を取得する
        ascending: Trueの場合は昇順（最小値）、Falseの場合は降順（最大値）で並び替え

    戻り値:
        dv1_x, dv1_y, dv, time_cost: n番目の行に対応するデータ
    """
    # CSVファイルを読み込む
    data = pd.read_csv(filename)

    # 'dv'列で並び替え（昇順または降順）
    sorted_data = data.sort_values(by='dv', ascending=ascending)

    # n番目の行を取得
    return sorted_data.iloc[n - 1][['dv1_x', 'dv1_y', 'dv', 'time_cost']].values

# 1番目からn番目までの情報をリストとして返す関数
def get_top_n(filename, n, ascending=True):
    """
    指定されたファイルから、1番目からn番目までのデータを取得する関数。

    引数:
        filename: CSVファイルのパス
        n: 取得する範囲（1番目からn番目まで）
        ascending: Trueの場合は昇順、Falseの場合は降順で取得
    戻り値:
        1番目からn番目までのdv1_x, dv1_y, dv, time_costを含むリスト
    """
    # CSVファイルを読み込む
    data = pd.read_csv(filename)
    # 'dv'列で並び替え
    sorted_data = data.sort_values(by='dv', ascending=ascending)
    # 1番目からn番目までのデータをリストとして返す
    return sorted_data.iloc[:n][['dv1_x', 'dv1_y', 'dv', 'time_cost']].values.tolist()


# Unscented Transform
def unscented_transform_m2e(mu, cov, f, x1, y1, r_aim, alpha=1e-3, kappa=2, beta=0):
    """
    mu: 平均ベクトル (入力)
    cov: 共分散行列 (入力)
    f: 適用する非線形関数
    alpha, kappa, beta: スケーリングパラメータ
    """
    n = len(mu)

    # シグマポイント生成
    sigma_points, lambda_ = ut.generate_sigma_points(mu, cov, alpha, kappa, beta)

    # 重みの計算
    Wm = np.zeros(2 * n + 1)
    Wc = np.zeros(2 * n + 1)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2 * n + 1):
        Wm[i] = Wc[i] = 1 / (2 * (n + lambda_))
        print(f"Wm{i}: {Wm[i]}")

    # シグマポイントに関数 f を適用
    # シグマポイントに関数 f を適用
    y = np.zeros(2 * n + 1)
    for i, sigma_point in enumerate(sigma_points):
        dv1 = [float(sigma_point[0]), float(sigma_point[1]), 0]
        dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost = f(x1, y1, r_aim, dv1)
        y[i] = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)
        print((f"y{i}: {y[i]}"))

    # 出力の平均
    mean_y = np.sum(Wm * y)

    # 出力の分散
    diff_y = y - mean_y
    cov_y = np.sum(Wc * (diff_y**2))

    return mean_y, cov_y, dv1_ans, dv2_ans


# 使用例：1番目からn番目までのデータを取得して表示
filename = 'gs_results.csv'  # CSVファイル名
n = 6  # 3番目まで取得
ascending = True  # Trueなら最小値順、Falseなら最大値順

R = 6378.137  # Earth radius
x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]  # 3000km
y1 = [384400, 0, 0, 0, 1.022, 0]
r_aim = R + 35786

# 1番目からn番目までのデータを取得して表示
top_n_data = get_top_n(filename, n, ascending)
for i, (dv1_x, dv1_y, dv, time_cost) in enumerate(top_n_data, start=1):

    # 各初期値dv1にガウスノイズを加える
    mu = np.array([dv1_x, dv1_y])
    cov = np.array([[1e-4, 0], [0, 1e-4]])

    mean, cov, dv1, dv2 = unscented_transform_m2e(mu, cov, tb.hohman_orbit3, x1, y1, r_aim)
    print(f"\n{i}番目の初期値 -> dv1_x: {dv1_x}, dv1_y: {dv1_y}")
    print(f"{i}番目の入力 -> dv(ノミナル): {dv}")
    print(f"{i}番目の出力 -> dv(平均値): {mean}, dv_cov(分散): {cov}")
    print(f"{i}番目のdv1 -> {dv1}, dv2 -> {dv2}")