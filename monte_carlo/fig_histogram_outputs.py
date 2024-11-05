import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ファイル名の指定
n = 1  # 適切な整数を指定してください
results_file = f"monte_carlo_results_{n}.csv"
summary_file = "monte_carlo_summary.csv"

# CSVファイルの読み込み
data_results = pd.read_csv(results_file)
data_summary = pd.read_csv(summary_file)

# outputのデータを取得
output_results = data_results['output']

# summaryから対応するnの行を取得
summary_row = data_summary[data_summary['n'] == n]
nominal_value = summary_row['nominal_value'].values[0]

# ヒストグラムの作成
plt.figure(figsize=(6, 5))  # プロットのサイズを小さくする
plt.hist(output_results, bins=30, alpha=0.7, color='g', edgecolor='black')

# 平均値と分散を計算し表示
mean_output = np.mean(output_results)
variance_output = np.var(output_results)
plt.axvline(mean_output, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_output:.2f}')
plt.axvline(mean_output + np.sqrt(variance_output), color='orange', linestyle='dotted', label=f'Std Dev: {np.sqrt(variance_output):.2f}')
plt.axvline(mean_output - np.sqrt(variance_output), color='orange', linestyle='dotted')

# ノミナル値を追加
plt.axvline(nominal_value, color='blue', linestyle='solid', label=f'Nominal: {nominal_value:.2f}')

# ヒストグラムの装飾
plt.title(f'Output Histogram (n={n})')
plt.xlabel('Output')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.show()
