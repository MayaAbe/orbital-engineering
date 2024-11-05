import pandas as pd
import core.two_body as tb
import stochastic.stochastic_o_1d as st

# CSVファイルの読み込み（ファイル名を指定）
df = pd.read_csv('gs_results.csv')

# 'dv'列でソートして、上位n行を取得
n = 5  # 取得する行数を指定
sorted_df = df.sort_values(by='dv').head(n)

# 各行ごとに変数に格納する
for i, row in enumerate(sorted_df.itertuples(index=False), start=1):
    globals()[f'row_{i}'] = [row.dv1_x, row.dv1_y, row.dv, row.time_cost]

# 出力確認
for i in range(1, n+1):
    print(f'row_{i}:', globals()[f'row_{i}'])

# ここで適当な2変数関数を定義
def f(dv1_x, dv1_y):
    return dv1_x**2 + dv1_y**2  # 例として2変数の二乗和を返す関数

R = 6378.137  # Earth radius
x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]  # 3000km
y1 = [384400, 0, 0, 0, 1.022, 0]

# それぞれの配列の dv1_x, dv1_y を関数に渡し、結果を出力する
for i in range(1, n+1):
    dv1_x, dv1_y, dv, time_cost = globals()[f'row_{i}']  # それぞれの要素を取得
    #result = f(dv1_x, dv1_y)  # 関数 f に dv1_x, dv1_y を引数として渡す
    if i == 1:
        st.add_gaussian_noise_1d(dv1_x)
        st.add_gaussian_noise_1d(dv1_y)
        dv1 = [dv1_x, dv1_y]
        result1, result2 = tb.hohman_orbit3(x1, y1, r_aim, dv1)
        abs_dv1 = np.linalg.norm(result1)
        abs_dv2 = np.linalg.norm(result2)
        result = abs_dv1 + abs_dv2
    print(f'Result for row_{i}: f({dv1_x}, {dv1_y}) = {result}')
