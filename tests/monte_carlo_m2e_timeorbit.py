import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import core.two_body as tb  # 指定のモジュールをインポート

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

# 軌道誤差を評価する関数
def evaluate_orbit_error(soltr2_nominal, soltr2_noisy):
    """
    soltr2とsoltr2_noisyの位置ベクトルの差を計算し、その差のノルムを合計して軌道誤差を評価します。
    """
    # 行数を揃えるための補完
    len_nominal = len(soltr2_nominal)
    len_noisy = len(soltr2_noisy)

    if len_nominal != len_noisy:
        if len_nominal > len_noisy:
            # noisy を nominal に合わせて補完
            interp_func = interp1d(
                np.linspace(0, 1, len_noisy),
                soltr2_noisy,
                axis=0,
                fill_value="extrapolate"
            )
            soltr2_noisy = interp_func(np.linspace(0, 1, len_nominal))
        else:
            # nominal を noisy に合わせて補完
            interp_func = interp1d(
                np.linspace(0, 1, len_nominal),
                soltr2_nominal,
                axis=0,
                fill_value="extrapolate"
            )
            soltr2_nominal = interp_func(np.linspace(0, 1, len_noisy))

    # 位置ベクトルの差を計算
    position_diff = soltr2_nominal[:, :3] - soltr2_noisy[:, :3]
    position_diff_norm = np.linalg.norm(position_diff, axis=1)

    # 位置差のノルムを合計
    orbit_error = np.sum(position_diff_norm)

    return float(orbit_error)

# モンテカルロシミュレーションを実行する関数
def monte_carlo_simulation_with_tb(filename, n, n_samples, noise_std_x, noise_std_y, results_file, summary_file):
    """
    モンテカルロシミュレーションを実行し、結果をCSVファイルに書き込みます。
    """
    # CSVファイルからdv1_x, dv1_y, dv, time_costを取得
    dv1_x, dv1_y, dv_total_nominal, time_cost_nominal = dv_minmax(filename, n)

    # 固定値を設定
    x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]
    y1 = [384400, 0, 0, 0, 1.022, 0]
    R = 6371  # 地球半径
    r_aim = R + 35786  # 静止軌道高度

    # ノミナル値の計算（ノイズがない場合）
    dv1_nominal = [dv1_x, dv1_y, 0]
    dv1_ans_nominal, dv2_ans_nominal, _, soltr_nominal, soltr2_nominal = tb.hohman_orbit3(
        x1, y1, r_aim, dv1_nominal, True, True
    )
    dv_nominal = np.linalg.norm(dv1_ans_nominal) + np.linalg.norm(dv2_ans_nominal)

    # 結果を保存するリスト
    results = []

    # サンプルごとの計算
    for i in range(n_samples):
        # 進捗状況を表示
        print(f"Sample {i+1}/{n_samples} for n={n}")

        # ガウスノイズをdv1に追加
        noisy_dv1_x = dv1_x + np.random.normal(0, noise_std_x)
        noisy_dv1_y = dv1_y + np.random.normal(0, noise_std_y)

        # ノイズを加えたdv1を作成
        dv1_noisy = [noisy_dv1_x, noisy_dv1_y, 0]

        # hohman_orbit3関数にノイズを加えたdv1を代入し出力を計算
        dv1_ans, dv2_ans, time_cost_sample, soltr, soltr2 = tb.hohman_orbit3(
            x1, y1, r_aim, dv1_noisy, True, True
        )

        if dv2_ans[0] is None:
            continue  # 無効値の場合はスキップ

        # dv_outputを計算（ノイズ後のdv1とdv2のノルムの和）
        dv_output = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)

        # 軌道誤差を評価
        # dv2にノイズを追加
        dv2_x_noisy = dv2_ans[0] + np.random.normal(0, noise_std_x)
        dv2_y_noisy = dv2_ans[1] + np.random.normal(0, noise_std_y)
        dv2_noisy = [dv2_x_noisy, dv2_y_noisy, dv2_ans[2]]  # z成分はそのまま

        # soltr2_noisyを計算
        r = [
            soltr[-1, 0],
            soltr[-1, 1],
            soltr[-1, 2],
            soltr[-1, 3] + dv2_noisy[0],
            soltr[-1, 4] + dv2_noisy[1],
            soltr[-1, 5] + dv2_noisy[2],
        ]
        t_span = [0, soltr2_nominal[-1, -1]]  # ノミナル軌道の終了時間に合わせる
        soltr2_noisy, _ = tb.MoonEarthSat(r, y1, 1, 300)

        orbit_error = evaluate_orbit_error(soltr2_nominal, soltr2_noisy)

        # 結果をリストに追加
        results.append({
            'dv1_x': dv1_noisy[0],
            'dv1_y': dv1_noisy[1],
            'dv_output': dv_output,
            'time_cost': time_cost_sample,
            'orbit_error': orbit_error
        })

    # 個別結果をDataFrameに変換
    df_results = pd.DataFrame(results)

    # 個別結果をCSVファイルに保存
    df_results.to_csv(results_file, index=False)

    # サマリー情報を計算
    dv_expected = df_results['dv_output'].mean()
    dv_variance = df_results['dv_output'].var()
    time_cost_expected = df_results['time_cost'].mean()
    time_cost_variance = df_results['time_cost'].var()
    orbit_matching_expect = df_results['orbit_error'].mean()
    orbit_matching_variance = df_results['orbit_error'].var()

    # サマリー結果を辞書で作成
    summary = {
        'n': n,
        'dv1_x': dv1_x,
        'dv1_y': dv1_y,
        'dv_nominal': dv_nominal,
        'dv_expected': dv_expected,
        'dv_variance': dv_variance,
        'time_cost_nominal': time_cost_nominal,
        'time_cost_expected': time_cost_expected,
        'time_cost_variance': time_cost_variance,
        'orbit_matching_expect': orbit_matching_expect,
        'orbit_matching_variance': orbit_matching_variance
    }

    # サマリーファイルに追記
    if not summary_file.exists():
        # ファイルが存在しない場合はヘッダーを書き込む
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_file, index=False)
    else:
        # ファイルが存在する場合は追記
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_file, mode='a', header=False, index=False)

if __name__ == '__main__':
    # パラメータ設定
    filename = 'gs_results.csv'  # CSVファイルのパスe
    i = 64  # nの開始値
    j = 70   # nの終了値
    n_samples = 1000  # モンテカルロシミュレーションのサンプル数
    noise_std_x = 0.001  # dv1_xのガウスノイズの標準偏差
    noise_std_y = 0.001  # dv1_yのガウスノイズの標準偏差

    # monte_carloディレクトリの作成
    monte_carlo_dir = Path.cwd() / 'monte_carlo11'
    monte_carlo_dir.mkdir(exist_ok=True)

    summary_file = monte_carlo_dir / 'monte_carlo_summary_time.csv'  # サマリーファイル

    # nをiからjまで変化させてモンテカルロシミュレーションを実行
    for n in range(i, j + 1):
        results_file = monte_carlo_dir / f'monte_carlo_results_time_{n}.csv'  # 各nの結果ファイル
        print(f"Running simulation for n={n}")
        monte_carlo_simulation_with_tb(filename, n, n_samples, noise_std_x, noise_std_y, results_file, summary_file)
