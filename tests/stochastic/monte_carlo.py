import pandas as pd
import numpy as np

# CSVファイルからdv1_x, dv1_y, dv, time_costを取得する関数
def dv_minmax(filename, n, ascending=True):
    """
    指定されたCSVファイルから、dv列を基準に並び替えたデータのn番目の情報を取得する関数。
    """
    # CSVファイルを読み込む
    data = pd.read_csv(filename)

    # 'dv'列で並び替え（昇順または降順）
    sorted_data = data.sort_values(by='dv', ascending=ascending)

    # n番目の行を取得
    return sorted_data.iloc[n - 1][['dv1_x', 'dv1_y', 'dv', 'time_cost']].values

# 2入力1出力の関数 f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# モンテカルロシミュレーションを実行する関数
def monte_carlo_simulation_with_noise(filename, n, n_samples, noise_std):
    """
    ガウスノイズを加えた入力に対するモンテカルロシミュレーション。
    """
    # CSVファイルからdv1_x, dv1_yを取得
    dv1_x, dv1_y, dv, time_cost = dv_minmax(filename, n)

    # 出力を保存するためのリスト
    outputs = []

    # ノミナル値の計算（ノイズがない場合）
    nominal_value = f(dv1_x, dv1_y)

    # 指定されたサンプル数だけモンテカルロシミュレーションを実行
    for _ in range(n_samples):
        # ガウスノイズを各入力に追加
        noisy_x = dv1_x + np.random.normal(0, noise_std)
        noisy_y = dv1_y + np.random.normal(0, noise_std)

        # 関数f(x, y)にノイズを加えた入力を代入し出力を計算
        output = f(noisy_x, noisy_y)
        outputs.append(output)

    # 出力の期待値（平均）を計算
    expected_value = np.mean(outputs)

    # 出力の分散を計算
    variance = np.var(outputs)

    return dv1_x, dv1_y, nominal_value, expected_value, variance

# パラメータ設定
filename = 'gs_results.csv'  # CSVファイルのパス
n = 1  # n番目のデータを取得
n_samples = 10000  # モンテカルロシミュレーションのサンプル数
noise_std = 0.1  # ガウスノイズの標準偏差

# モンテカルロシミュレーションの実行
dv1_x, dv1_y, nominal_value, expected_value, variance = monte_carlo_simulation_with_noise(filename, n, n_samples, noise_std)

# 結果を出力
print(f"入力値 dv1_x: {dv1_x}, dv1_y: {dv1_y}")
print(f"ノイズがない場合の出力 (ノミナル値): {nominal_value}")
print(f"出力の期待値（ノイズあり）: {expected_value}")
print(f"出力の分散（ノイズあり）: {variance}")
