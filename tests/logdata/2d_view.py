import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# カレントディレクトリのパスを取得
current_dir = Path.cwd()

# カレントディレクトリ内のCSVファイルのパス指定（Pathlibを使用）
csv_file = current_dir / 'grid_search_results_detail_x080_090_ym2.7_m2.65.csv'

# CSVファイルの読み込み
df = pd.read_csv(csv_file)

# dvがinfかどうかのフラグを作成
df['dv_is_inf'] = np.isinf(df['dv'])

# dvが10以下かどうかのフラグを作成
df['dv_less_than_10'] = df['dv'] <= 10

# dvが10以下かつinfでないデータと、その他のデータに分ける
df_gradation = df[df['dv_less_than_10'] & ~df['dv_is_inf']]
df_red = df[~df['dv_less_than_10'] | df['dv_is_inf']]

# プロットの作成
plt.figure(figsize=(10, 6))

# マーカーサイズを小さくする (s=10)
# dvが10以下のデータを散布図でプロット（色の濃淡で表現）
plt.scatter(df_gradation['dv1_x'], df_gradation['dv1_y'], c=df_gradation['dv'], cmap='viridis', marker='o', label='dv <= 10', s=10)

# dvが10を超えるデータ、またはinfのデータを赤でプロット
plt.scatter(df_red['dv1_x'], df_red['dv1_y'], color='red', marker='x', label='dv  = inf', s=10)

# カラーバーの表示（dvが10以下のデータに対して）
plt.colorbar(label='dv (<= 10)')

# 軸ラベルとタイトル
plt.xlabel('dv1_x')
plt.ylabel('dv1_y')
plt.title('Plot of dv1_x vs dv1_y with dv color scale')

# 凡例を追加
plt.legend()

# プロットの表示
plt.show()
