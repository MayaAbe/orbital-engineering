# main.py

from functions import sphere_function, rastrigin_function
from genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmHohmann
import matplotlib.pyplot as plt
import numpy as np
# 必要に応じて以下のモジュールをインポート
# import core.two_body as tb
# import core.orbit_calc as oc

def optimize_function():
    # 最適化する関数を選択
    func = sphere_function  # 他の関数を使用する場合はここを変更

    # 各変数の範囲を定義（例：[-5.12, 5.12]の範囲で2次元）
    bounds = [(-5.12, 5.12) for _ in range(2)]

    # 遺伝的アルゴリズムのパラメータを設定
    ga = GeneticAlgorithm(
        func=func,
        bounds=bounds,
        population_size=100,
        generations=200,
        mutation_rate=0.01,
        crossover_rate=0.7
    )

    # アルゴリズムを実行
    best_individual, best_fitness, fitness_history = ga.run()

    # 結果を表示
    print("最適解:", best_individual)
    print("最適値:", best_fitness)

    # 適応度の履歴をプロット
    plt.figure()
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Evolution for Function Optimization')
    plt.grid()
    plt.show()

def optimize_hohmann():
    # 長さの単位はkm, 時間の単位はs

    R = 6378.137  # Earth radius
    x1 = [384400 + 3000, 0, 0, 0, 1.022 + 1.02, 0]  # 3000km
    y1 = [384400, 0, 0, 0, 1.022, 0]

    # 探索する変数の範囲を定義
    bounds = [
        (0.82, 0.84),    # dv1のx成分の範囲
        (-2.7, -2.67)    # dv1のy成分の範囲
    ]

    # 遺伝的アルゴリズムのパラメータを設定
    ga = GeneticAlgorithmHohmann(
        x1=x1,
        y1=y1,
        r_aim=R + 35786,
        bounds=bounds,
        population_size=10,
        generations=2,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

    # アルゴリズムを実行
    result = ga.run()

    # 結果を表示
    print("最適な個体:", result['best_individual'])
    print("最適な適応度（dvの総和）:", result['best_fitness'])
    print("dv1_ans:", result['dv1_ans'])
    print("dv2_ans:", result['dv2_ans'])

    # 軌道をプロット
    sol1 = result['sol1']
    sol2 = result['sol2']
    sol_com = result['sol_com']

    plt.plot(sol1[:, 0], sol1[:, 1], 'b', label='before dv1')
    plt.plot(sol_com[:, 0], sol_com[:, 1], 'k', label='trajectory')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r--', label='target orbit')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Trajectory optimized by Genetic Algorithm')
    plt.show()

    # 適応度の履歴をプロット
    plt.figure()
    plt.plot(result['fitness_history'])
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Total dv)')
    plt.title('Fitness Evolution for Hohmann Transfer')
    plt.grid()
    plt.show()

def main():
    # functions.py 内の関数を最適化
    # optimize_function()

    # hohman_orbit3 の最適化を実行
    optimize_hohmann()

if __name__ == "__main__":
    main()
