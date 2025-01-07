import pandas as pd

# ファイルを読み込み、共分散を計算して新しい列を追加する関数
def add_covariance_column(summary_file):
    """
    monte_carlo_summary.csvファイルを読み込み、dv_varianceとtime_cost_varianceから共分散を計算し、新しい列を追加。
    """
    # CSVファイルの読み込み
    df = pd.read_csv(summary_file)

    # 共分散を計算
    df['covariance'] = df[['dv_variance', 'time_cost_variance']].cov().iloc[0, 1]

    # 共分散列を追加してファイルに上書き保存
    df.to_csv(summary_file, index=False)
    print(f"共分散列を追加して {summary_file} に保存しました。")

# ファイルパスを指定
summary_file = 'monte_carlo\zmonte_carlo_summary_time.csv'

# 関数を実行
add_covariance_column(summary_file)
