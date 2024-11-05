import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# CSVファイルのパスを設定
csv_path = os.path.join('monte_carlo', 'monte_carlo_summary.csv')

# CSVファイルを読み込む
data = pd.read_csv(csv_path)

# 目的関数としての列を取得（値が小さいほど良い解）
objectives = data[['expected_value', 'variance', 'time_cost']].values

# パレート効率のある点を判定する関数
def is_pareto_efficient(costs):
    """
    パレート効率のある点を判定する。
    入力:
        costs - 各目的関数の値の配列（形状: [n_samples, n_objectives]）
    出力:
        is_efficient - パレート効率のある点のブールマスク（形状: [n_samples]）
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # 次にチェックする点のインデックス
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # 劣っている点を除外
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index+1])
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask

# パレート効率のある点を判定
pareto_mask = is_pareto_efficient(objectives)

# パレート最適解を抽出
pareto_front = data[pareto_mask]

# パレート最適解を配列として表示
print("[expected_value, variance, time_cost] ")
print(pareto_front[['expected_value', 'variance', 'time_cost']].values)

# 3次元プロットによる可視化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 全データ点を薄い色でプロット
ax.scatter(data['expected_value'], data['variance'], data['time_cost'], c='lightgray', label='All Solutions')

# パレート最適解を赤色でプロット
ax.scatter(pareto_front['expected_value'], pareto_front['variance'], pareto_front['time_cost'], c='red', label='Pareto Front')

# 軸ラベルの設定（英語）
ax.set_xlabel('Expected Value')
ax.set_ylabel('Variance')
ax.set_zlabel('Time Cost')

# タイトルと凡例の設定（英語）
ax.set_title('Pareto Front in 3D Space')
ax.legend()

# プロットの表示
plt.show()
