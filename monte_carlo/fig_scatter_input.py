import pandas as pd
import matplotlib.pyplot as plt

# ファイル名の指定
n = 1  # 適切な整数を指定してください
results_file = f"monte_carlo_results_{n}.csv"
summary_file = "monte_carlo_summary.csv"
output_threshold = 4.0  # 閾値を設定（例として10）

# CSVファイルの読み込み
data_results = pd.read_csv(results_file)
data_summary = pd.read_csv(summary_file)

# dv1_x, dv1_y, outputのデータを取得
dv1_x_results = data_results['dv1_x']
dv1_y_results = data_results['dv1_y']
output_results = data_results['output']

# summaryから対応するnの行を取得
summary_row = data_summary[data_summary['n'] == n]
nominal_dv1_x = summary_row['dv1_x'].values[0]
nominal_dv1_y = summary_row['dv1_y'].values[0]

# 閾値以上の出力を持つデータ（unstable output）の抽出
unstable_mask = output_results >= output_threshold
stable_mask = ~unstable_mask

# 散布図の作成（ノイズが加わったデータ、unstable output、ノイズがないデータ）
plt.figure(figsize=(6, 5))  # プロットのサイズを小さくする
# 安定しているデータ（青）
plt.scatter(dv1_x_results[stable_mask], dv1_y_results[stable_mask], alpha=0.4, color='b', label='Noisy Input(Zone a, b)')

# 不安定なデータが存在する場合のみオレンジのプロットを追加
if unstable_mask.sum() > 0:
    plt.scatter(dv1_x_results[unstable_mask], dv1_y_results[unstable_mask], alpha=0.4, color='orange', label='Unstable Noisy Input(Zone c)')

# ノイズがないデータ（赤）
plt.scatter(nominal_dv1_x, nominal_dv1_y, color='r', s=100, label='Nominal Input')

# 散布図の装飾
plt.title(f'Scatter plot of dv1_x vs dv1_y (n={n})')
plt.xlabel('dv1_x')
plt.ylabel('dv1_y')
plt.grid(True)
plt.legend()
plt.show()
