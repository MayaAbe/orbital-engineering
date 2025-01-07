import pandas as pd
import matplotlib.pyplot as plt

def plot_dv_output_vs_time_cost(n, output_threshold):
    """
    指定したファイル zmonte_carlo_results_time_{n}.csv から dv_output と time_cost をプロットする。

    Parameters:
    n (int): ファイルの番号
    output_threshold (float): dv_output の閾値
    """
    # ファイル名の作成
    file_name = f"zmonte_carlo_results_time_{n}.csv"

    try:
        # CSVファイルの読み込み
        data = pd.read_csv(file_name)

        # 閾値以上のデータを抽出
        filtered_data = data[data['dv_output'] >= output_threshold]

        # dv_output と time_cost のプロット
        plt.figure(figsize=(8, 6))
        plt.scatter(filtered_data['dv_output'], filtered_data['time_cost'], alpha=0.6, color='blue', label=f'dv_output >= {output_threshold}')
        plt.title(f'Relationship between dv_output and time_cost (n={n})')
        plt.xlabel('dv_output')
        plt.ylabel('time_cost')
        plt.grid(True)
        plt.legend()
        plt.show()
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except KeyError as e:
        print(f"Column {e} missing in the file {file_name}.")

# 使用例
n = 3  # 任意のファイル番号
output_threshold = 3.8  # 閾値を設定
plot_dv_output_vs_time_cost(n, output_threshold)
