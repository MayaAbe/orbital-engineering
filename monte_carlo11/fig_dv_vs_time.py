import pandas as pd
import matplotlib.pyplot as plt

# CSVファイル名
filename = 'monte_carlo_summary_time.csv'

# CSVファイルの読み込み
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"{filename} が見つかりません。カレントディレクトリに存在することを確認してください。")
    exit()

# グラフの設定
plt.figure(figsize=(12, 10))

# 1. dv_nominal と time_cost_nominal
plt.subplot(3, 1, 1)
plt.scatter(df['dv_nominal'], df['time_cost_nominal'], c='blue', alpha=0.7)
plt.xlabel('dv_nominal')
plt.ylabel('time_cost_nominal')
plt.title('dv_nominal vs time_cost_nominal')
plt.grid(True)

# 2. dv_expected と time_cost_expected
plt.subplot(3, 1, 2)
plt.scatter(df['dv_expected'], df['time_cost_expected'], c='green', alpha=0.7)
plt.xlabel('dv_expected')
plt.ylabel('time_cost_expected')
plt.title('dv_expected vs time_cost_expected')
plt.grid(True)

# 3. dv_variance と time_cost_variance（横軸を対数表示）
plt.subplot(3, 1, 3)
plt.scatter(df['dv_variance'], df['time_cost_variance'], c='red', alpha=0.7)
plt.xlabel('dv_variance (log scale)')
plt.ylabel('time_cost_variance')
plt.title('dv_variance vs time_cost_variance (Log Scale)')
plt.xscale('log')  # 横軸を対数スケールに設定
plt.grid(True)

# レイアウト調整と表示
plt.tight_layout()
plt.show()
