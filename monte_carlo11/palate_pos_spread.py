import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# ファイルパスのリストを取得（"dataset_1.csv", "dataset_2.csv", ...）
file_list = sorted(glob.glob("monte_carlo_results_time_*.csv"))

results = []

for file in file_list:
    df = pd.read_csv(file)
    # 各データセット内には、dv_outputとtime_costの列があると仮定
    dv = df["dv_output"].values
    time = df["time_cost"].values

    # 平均値(期待値)と分散を算出
    E_dv = np.mean(dv)
    Var_dv = np.var(dv, ddof=1)  # 不偏分散を用いる場合ddof=1
    E_time = np.mean(time)
    Var_time = np.var(time, ddof=1)

    # PosとSpreadを計算
    # Pos = sqrt(E(dv)^2 + E(time)^2)
    Pos = np.sqrt((E_dv**2) + (E_time**2))
    # Spread = Var(dv) + Var(time)
    Spread = Var_dv + Var_time

    # 結果を格納
    results.append([file, E_dv, E_time, Var_dv, Var_time, Pos, Spread])

# 結果をDataFrameとして整理
res_df = pd.DataFrame(results, columns=["filename", "E(dv)", "E(time)", "Var(dv)", "Var(time)", "Pos", "Spread"])

# パレートフロント抽出
# 2目的(Pos, Spread)をともに最小化と考え、非劣解（Pareto front）を求める関数を定義
def pareto_front(points):
    # points: shape (N, 2) の array
    # 両方とも「小さいほうが良い」目的とする
    is_non_dominated = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if i != j:
                # jがiを支配していれば、iは非劣解でない
                if (points[j,0] <= points[i,0]) and (points[j,1] <= points[i,1]) and ((points[j,0] < points[i,0]) or (points[j,1] < points[i,1])):
                    is_non_dominated[i] = False
                    break
    return is_non_dominated

points = res_df[["Pos", "Spread"]].values
is_non_dominated = pareto_front(points)

# 可視化
plt.figure(figsize=(8,6))
plt.scatter(res_df["Pos"], res_df["Spread"], c="blue", label="All Solutions")
plt.scatter(res_df.loc[is_non_dominated, "Pos"],
            res_df.loc[is_non_dominated, "Spread"],
            c="red", label="Pareto Front")
plt.xlabel("Pos (smaller is better)")
plt.ylabel("Spread (smaller is better)")
plt.title("2D Objective Space (Pos vs Spread) with Pareto Front")
plt.legend()
plt.grid(True)
plt.show()
