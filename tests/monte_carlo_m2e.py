import pandas as pd
import numpy as np
import core.two_body as tb  # 指定のモジュールをインポート
from pathlib import Path

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

# モンテカルロシミュレーションを実行する関数 (tb.hohman_orbit3関数を使用)
def monte_carlo_simulation_with_tb(filename, n, n_samples, noise_std_x, noise_std_y, results_file, summary_file):
    """
    tb.hohman_orbit3を用いたモンテカルロシミュレーション。ガウスノイズをdv1_xとdv1_yにそれぞれ追加。
    """
    # CSVファイルからdv1_x, dv1_y, dv, time_costを取得
    dv1_x, dv1_y, dv, time_cost = dv_minmax(filename, n)

    # 固定値を設定
    x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]  # 固定値
    y1 = [384400, 0, 0, 0, 1.022, 0]  # 固定値
    R = 6371  # 地球半径
    r_aim = R + 35786  # geostationary orbit altitude

    # 出力を保存するためのリスト
    outputs = []
    input_output_data = []

    # ノミナル値の計算（ノイズがない場合）
    dv1_nominal = [dv1_x, dv1_y, 0]
    dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost0 = tb.hohman_orbit3(x1, y1, r_aim, dv1_nominal)
    nominal_value = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)

    # 指定されたサンプル数だけモンテカルロシミュレーションを実行
    for i in range(n_samples):
        print(f"Sample {i+1}/{n_samples} for n={n}")
        # ガウスノイズを各入力に追加
        noisy_dv1_x = dv1_x + np.random.normal(0, noise_std_x)
        noisy_dv1_y = dv1_y + np.random.normal(0, noise_std_y)

        # ノイズを加えたdv1を作成
        dv1 = [noisy_dv1_x, noisy_dv1_y, 0]

        # tb.hohman_orbit3関数にノイズを加えた入力を代入し出力を計算
        dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost1 = tb.hohman_orbit3(x1, y1, r_aim, dv1)
        if dv2_ans[0] is None:
            dv2_ans = [np.inf, np.inf, np.inf]
        # dv1_ansとdv2_ansの絶対値の和を出力として保存
        output = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)
        outputs.append(output)

        # 各ループの入力値と出力値を保存
        input_output_data.append([noisy_dv1_x, noisy_dv1_y, output])

    # 結果をCSVに保存
    results_df = pd.DataFrame(input_output_data, columns=["dv1_x", "dv1_y", "output"])
    results_df.to_csv(results_file, index=False)

    # 出力の期待値（平均）を計算
    expected_value = np.mean(outputs)

    # 出力の分散を計算
    variance = np.var(outputs)
    nominal_value = dv

    # サマリーを追記保存
    summary_data = {
        "n": n,
        "dv1_x": dv1_x,
        "dv1_y": dv1_y,
        "nominal_value": nominal_value,
        "expected_value": expected_value,
        "variance": variance,
        "time_cost": time_cost  # 末尾に配置されるよう指定
    }

    # 列順を指定してDataFrameに変換
    summary_df = pd.DataFrame([summary_data], columns=["n", "dv1_x", "dv1_y", "nominal_value", "expected_value", "variance", "time_cost"])

    # サマリーファイルに追記
    if not summary_file.exists():
        summary_df.to_csv(summary_file, index=False)
    else:
        summary_df.to_csv(summary_file, mode='a', header=False, index=False)

    return dv1_x, dv1_y, nominal_value, expected_value, variance

if __name__ == '__main__':
    # パラメータ設定
    filename = 'gs_results.csv'  # CSVファイルのパス
    i = 71  # nの開始値
    j = 150  # nの終了値
    n_samples = 1000  # モンテカルロシミュレーションのサンプル数
    noise_std_x = 0.001  # dv1_xのガウスノイズの標準偏差
    noise_std_y = 0.001  # dv1_yのガウスノイズの標準偏差

    # monte_carloディレクトリの作成
    monte_carlo_dir = Path.cwd() / 'monte_carlo'
    monte_carlo_dir.mkdir(exist_ok=True)

    summary_file = monte_carlo_dir / 'monte_carlo_summary.csv'  # サマリーファイル

    # nをiからjまで変化させてモンテカルロシミュレーションを実行
    for n in range(i, j + 1):
        results_file = monte_carlo_dir / f'monte_carlo_results_{n}.csv'  # 各nの結果ファイル
        print(f"Running simulation for n={n}")
        monte_carlo_simulation_with_tb(filename, n, n_samples, noise_std_x, noise_std_y, results_file, summary_file)
