import csv

def find_row_with_extreme_value(filename, column_name, rank=0, find_min=True):
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        rows = list(csv_reader)

        # Extract the target column values and sort them
        values = [(float(row[column_name]), row) for row in rows]
        values.sort(key=lambda x: x[0], reverse=not find_min)

        # Ensure the rank is within the bounds of the list
        if rank < 0 or rank >= len(values):
            raise ValueError("Rank is out of bounds")

        # Get the target row based on the rank
        target_value, target_row = values[rank]
        return target_row

def main():
    filename = input("CSVファイルの名前を入力してください: ")

    while True:
        column_name = input("検索する列名を入力してください: ")
        rank = int(input("最小値または最大値から何番目の値を探しますか (0が最小/最大、1が2番目、2が3番目...): "))
        find_min_str = input("最小値から探しますか？ (yes/no): ")
        find_min = find_min_str.lower() == 'yes'

        try:
            result_row = find_row_with_extreme_value(filename, column_name, rank, find_min)
            print("結果の行: ", result_row)
        except ValueError as e:
            print("エラー: ", e)
        except FileNotFoundError:
            print("ファイルが見つかりませんでした。ファイル名を確認してください。")
        except KeyError:
            print("指定された列が見つかりません。列名を確認してください。")
        except Exception as e:
            print("予期しないエラーが発生しました: ", e)

        continue_search = input("引き続き検索をしますか？ (yes/no): ").lower()
        if continue_search != 'yes':
            break

if __name__ == "__main__":
    main()
