import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
summary_file_path = 'monte_carlo_summary.csv'
summary_data = pd.read_csv(summary_file_path)

# データを抽出
n_values = summary_data['n']
dv1_x_values = summary_data['dv1_x']
dv1_y_values = summary_data['dv1_y']

# 散布図の設定
plt.figure(figsize=(8, 6))

# 各nに対応する monte_carlo_results_{n}.csv を処理
for n in n_values:
    result_file_path = f'monte_carlo_results_{n}.csv'
    try:
        # monte_carlo_results_{n}.csv を読み込む
        result_data = pd.read_csv(result_file_path)

        # ノイズが加わった dv1_x と dv1_y をプロット
        noisy_dv1_x = result_data.iloc[:, 0]
        noisy_dv1_y = result_data.iloc[:, 1]

        # ノイズの加わった入力をプロット（小さく、半透明）
        plt.scatter(noisy_dv1_x, noisy_dv1_y, color='gray', alpha=0.1, s=10)

    except FileNotFoundError:
        print(f'Warning: {result_file_path} not found, skipping.')

# monte_carlo_summary.csvのデータをプロット
plt.scatter(dv1_x_values, dv1_y_values, color='red', s=20, label='Nominal Values')

# 各プロットの横にnの値を付す
for i, n in enumerate(n_values):
    plt.text(dv1_x_values[i], dv1_y_values[i], str(n), fontsize=9, ha='right')

# グラフのタイトルと軸ラベルを設定
plt.title('Scatter plot of dv1_x vs dv1_y with Nominal and Noisy Inputs')
plt.xlabel('dv1_x')
plt.ylabel('dv1_y')
plt.legend()

# グリッドを表示し、グラフを描画
plt.grid(True)
plt.tight_layout()
plt.show()
