import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms

# データ読み込み
df = pd.read_csv('gs_results.csv')

# time_costが数値であれば、そのまま扱う
df['time'] = df['time_cost']  # time_cost列をそのまま時間として扱う
df['cost'] = df['time_cost']  # ここは実際のデータ構造に基づき、修正が必要かもしれません
df['uncertainty'] = df['time_cost']  # 実際のデータ構造に基づき、修正が必要です

# DataFrameをリスト化してランダム選択できるようにする
df_list = list(df.iterrows())

# パレート最適化用に最小化を定義
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# データをもとに個体を作成
def create_individual(row):
    return creator.Individual([row['dv1_x'], row['dv1_y'], row['time'], row['cost'], row['uncertainty']])

# populationを生成する
toolbox.register("population", tools.initRepeat, list, lambda: create_individual(random.choice(df_list)[1]))

# 目的関数
def eval_problem(individual):
    time = individual[2]  # time
    cost = individual[3]  # cost
    uncertainty = individual[4]  # uncertainty
    return time, cost, uncertainty

toolbox.register("evaluate", eval_problem)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=50,
                              stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof

def plot_pareto_front(pop):
    # パレートフロントを抽出
    pareto_front = np.array([ind.fitness.values for ind in pop if ind.fitness.valid])

    # 3Dプロットのセットアップ
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # パレートフロントを3次元でプロット
    ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='b', marker='o')

    # 軸ラベルを設定
    ax.set_xlabel('Time')
    ax.set_ylabel('Cost')
    ax.set_zlabel('Uncertainty')

    plt.title('Pareto Front')
    plt.show()

if __name__ == "__main__":
    pop, stats, hof = main()
    plot_pareto_front(pop)
