import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 外れ値の表示を選択するフラグ（Trueで表示、Falseで非表示）
show_outliers = True

# monte_carlo_summary.csv の読み込み
summary_df = pd.read_csv('monte_carlo_summary.csv')

# グラフの作成
fig, ax = plt.subplots(figsize=(10, 6))

# nominal_value の折れ線グラフ
ax.plot(summary_df['n'], summary_df['nominal_value'], marker='o', linestyle='-', color='b', label='Nominal Value')

# expected_value の折れ線グラフ
ax.plot(summary_df['n'], summary_df['expected_value'], marker='x', linestyle='--', color='r', label='Expected Value')

# 各 n に対して対応する monte_carlo_results_{n}.csv を読み込み、箱ひげ図を作成
for i in range(len(summary_df)):
    n_value = summary_df['n'][i]

    # 対応するファイル名
    file_name = Path(f"monte_carlo_results_{n_value}.csv")

    if file_name.exists():
        # monte_carlo_results_{n}.csv の読み込み
        results_df = pd.read_csv(file_name)

        # output 列の期待値と分散を計算
        output_values = results_df['output']
        expected_val = output_values.mean()
        variance = output_values.var()

        # output の値に基づいて箱ひげ図を作成
        ax.boxplot(output_values, positions=[n_value], widths=0.3, patch_artist=False,
                   showfliers=show_outliers, flierprops=dict(marker='+', color='gray', markersize=8))

# グラフの装飾
ax.set_xlabel('n')
ax.set_ylabel('Values')
ax.set_title('Nominal Value, Expected Value, and Boxplots from monte_carlo_results_{n}.csv')
ax.legend()
ax.grid(True)

# グラフの表示
plt.show()
