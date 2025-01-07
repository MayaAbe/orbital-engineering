import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import mplcursors

# CSVファイルの一覧を取得
file_list = sorted(glob.glob("zmonte_carlo_results_time_*.csv"))

results = []

for file in file_list:
    df = pd.read_csv(file)
    dv = df["dv_output"].values
    time = df["time_cost"].values

    E_dv = np.mean(dv)
    Var_dv = np.var(dv, ddof=1)   # 不偏分散
    E_time = np.mean(time)
    Var_time = np.var(time, ddof=1)

    # Spread = 幾何平均に常用対数を取った値
    Spread = np.log10(np.sqrt(Var_dv * Var_time))

    # Var(dv), Var(time)も記録
    results.append([file, E_dv, E_time, Spread, Var_dv, Var_time])

res_df = pd.DataFrame(results, columns=["filename", "E(dv)", "E(time)", "Spread", "Var(dv)", "Var(time)"])

# 型変換
for col in ["E(dv)", "E(time)", "Spread", "Var(dv)", "Var(time)"]:
    res_df[col] = pd.to_numeric(res_df[col], errors='coerce')

# 有効データ抽出
res_df = res_df[np.isfinite(res_df["E(dv)"]) &
                np.isfinite(res_df["E(time)"]) &
                np.isfinite(res_df["Spread"]) &
                np.isfinite(res_df["Var(dv)"]) &
                np.isfinite(res_df["Var(time)"])]

if res_df.empty:
    print("No valid data.")
    exit()

def pareto_front(points):
    N = points.shape[0]
    is_non_dominated = np.ones(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j:
                if (points[j,0] <= points[i,0]) and (points[j,1] <= points[i,1]) and (points[j,2] <= points[i,2]) and \
                   ((points[j,0] < points[i,0]) or (points[j,1] < points[i,1]) or (points[j,2] < points[i,2])):
                    is_non_dominated[i] = False
                    break
    return is_non_dominated

points_2d = res_df[["E(dv)", "E(time)", "Spread"]].values
is_non_dominated = pareto_front(points_2d)

x_min, x_max = None, None
y_min, y_max = None, None

if x_min is None: x_min = res_df["E(dv)"].min()
if x_max is None: x_max = res_df["E(dv)"].max()
if y_min is None: y_min = res_df["E(time)"].min()
if y_max is None: y_max = res_df["E(time)"].max()

in_range_df = res_df[
    (res_df["E(dv)"] >= x_min) & (res_df["E(dv)"] <= x_max) &
    (res_df["E(time)"] >= y_min) & (res_df["E(time)"] <= y_max)
]

if in_range_df.empty:
    print("No data in the specified range.")
    exit()

in_range_points = in_range_df[["E(dv)", "E(time)", "Spread"]].values
in_range_is_non_dominated = pareto_front(in_range_points)

def finite_or_default(val, default=0):
    return val if (np.isfinite(val)) else default

x_min = finite_or_default(x_min, 0)
x_max = finite_or_default(x_max, x_min+1)
y_min = finite_or_default(y_min, 0)
y_max = finite_or_default(y_max, y_min+1)

non_pf_mask = ~in_range_is_non_dominated
non_pf_data = in_range_df.loc[non_pf_mask]
pf_data = in_range_df.loc[in_range_is_non_dominated]

fig, ax = plt.subplots(figsize=(8, 6))

# 非パレート解：グレー半透明
scatter_non_pf = ax.scatter(non_pf_data["E(dv)"], non_pf_data["E(time)"],
                            c="gray", s=20, alpha=0.3, edgecolors='none')

# パレート解：Spreadを等間隔で表現
scatter_pf = ax.scatter(pf_data["E(dv)"], pf_data["E(time)"],
                        c=pf_data["Spread"], cmap='coolwarm',
                        s=60, alpha=1.0, edgecolors='black', linewidths=0.5)

ax.set_xlabel("E(dv) (smaller is better)")
ax.set_ylabel("E(time) (smaller is better)")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title("2D Objective Space (E(dv), E(time))\nSpread(log10(Geometric Mean)) indicated by color")

# カラーバー
cbar = plt.colorbar(scatter_pf, ax=ax)
cbar.set_label("Spread")

# DataFrameをリセットしてインデックス対応を簡単に
pf_data_reset = pf_data.reset_index(drop=True)
non_pf_data_reset = non_pf_data.reset_index(drop=True)

def get_annotation_text(df, ind):
    row = df.iloc[ind]
    text = (f"E(dv): {row['E(dv)']:.4f}\n"
            f"E(time): {row['E(time)']:.4f}\n"
            f"Spread: {row['Spread']:.4f}\n"
            f"Var(dv): {row['Var(dv)']:.4f}\n"
            f"Var(time): {row['Var(time)']:.4f}")
    return text

# scatter_pfとscatter_non_pfをまとめてカーソルで管理
cursor = mplcursors.cursor([scatter_pf, scatter_non_pf], hover=True)

@cursor.connect("add")
def on_add(sel):
    if sel.artist == scatter_pf:
        df_to_use = pf_data_reset
    else:
        df_to_use = non_pf_data_reset
    sel.annotation.set_text(get_annotation_text(df_to_use, sel.index))
    sel.annotation.get_bbox_patch().set_facecolor('white')

plt.tight_layout()
plt.show()
