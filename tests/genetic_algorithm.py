# genetic_algorithm.py

import random
import numpy as np
import time
import matplotlib.pyplot as plt
import core.two_body as tb
import core.orbit_calc as oc

class GeneticAlgorithm:
    def __init__(self, func, bounds, population_size=100, generations=100, mutation_rate=0.01, crossover_rate=0.7):
        """
        遺伝的アルゴリズムの初期化

        :param func: 最適化する目的関数
        :param bounds: 各変数の範囲を示すタプルのリスト [ (下限, 上限), ... ]
        :param population_size: 個体群のサイズ
        :param generations: 世代数
        :param mutation_rate: 突然変異率
        :param crossover_rate: 交叉率
        """
        self.func = func
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self):
        """
        初期個体群の生成
        """
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dimensions)]
            population.append(individual)
        return population

    def fitness(self, individual):
        """
        適応度の計算（目的関数の値）
        """
        return self.func(individual)

    def selection(self, population):
        """
        選択：適応度に基づいて個体を選択
        """
        population = sorted(population, key=lambda x: self.fitness(x))
        return population[:int(0.5 * self.population_size)]  # 上位50%を選択

    def crossover(self, parent1, parent2):
        """
        交叉：2つの親から子を生成
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.dimensions - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        """
        突然変異：個体の一部をランダムに変更
        """
        for i in range(self.dimensions):
            if random.random() < self.mutation_rate:
                # 変数に対して小さな変化を加える（変異幅を制限）
                mutation_value = random.uniform(-0.1, 0.1)
                individual[i] += mutation_value
                # 範囲外に出ないように調整
                individual[i] = min(max(individual[i], self.bounds[i][0]), self.bounds[i][1])
        return individual

    def run(self):
        """
        遺伝的アルゴリズムの実行
        """
        population = self.initialize_population()
        best_individual = None
        best_fitness = float('inf')
        fitness_history = []

        for gen in range(self.generations):
            selected = self.selection(population)
            next_generation = []
            while len(next_generation) < self.population_size:
                parents = random.sample(selected, 2)
                offspring = self.crossover(parents[0], parents[1])
                next_generation.extend([self.mutate(child) for child in offspring])
            population = next_generation[:self.population_size]
            current_best = min(population, key=lambda x: self.fitness(x))
            current_fitness = self.fitness(current_best)
            fitness_history.append(current_fitness)
            if current_fitness < best_fitness:
                best_individual = current_best
                best_fitness = current_fitness
            # print(f"Generation {gen}: Best Fitness = {current_fitness}")

        return best_individual, best_fitness, fitness_history


# 以下に hohman_orbit3 の最適化に特化したクラスを追加
class GeneticAlgorithmHohmann:
    def __init__(self, x1, y1, r_aim, bounds, population_size=50, generations=50, mutation_rate=0.1, crossover_rate=0.7):
        """
        hohman_orbit3 の最適化に特化した遺伝的アルゴリズム

        :param x1: 出発点の状態ベクトル
        :param y1: 目的地の状態ベクトル
        :param r_aim: 目標半径
        :param bounds: 探索する変数の範囲 [ (dv1_x下限, dv1_x上限), (dv1_y下限, dv1_y上限) ]
        :param population_size: 個体群のサイズ
        :param generations: 世代数
        :param mutation_rate: 突然変異率
        :param crossover_rate: 交叉率
        """
        self.x1 = x1
        self.y1 = y1
        self.r_aim = r_aim
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self):
        """
        初期個体群の生成
        """
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dimensions)]
            population.append(individual)
        return population

    def fitness(self, individual):
        """
        適応度の計算：dvの総和を最小化する
        """
        dv1 = [individual[0], individual[1], 0]
        dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost = tb.hohman_orbit3(self.x1, self.y1, self.r_aim, dv1)

        # dv1_ans または dv2_ans が None の場合、最大の適応度を返す
        if dv1_ans is None or dv2_ans is None or dv2_ans[0] is None:
            return float('inf')  # 無効な解には最大の適応度を与える

        dv_total = np.linalg.norm(dv1_ans) + np.linalg.norm(dv2_ans)
        # 目標半径に到達していない場合はペナルティを与える
        if np.linalg.norm(sol_com[-1][0:3]) < self.r_aim:
            dv_total += 1000  # 大きなペナルティ

        return dv_total

    def selection(self, population):
        """
        選択：適応度に基づいて個体を選択
        """
        population = sorted(population, key=lambda x: self.fitness(x))
        return population[:int(0.5 * self.population_size)]  # 上位50%を選択

    def crossover(self, parent1, parent2):
        """
        交叉：2つの親から子を生成
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.dimensions - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        """
        突然変異：個体の一部をランダムに変更
        """
        for i in range(self.dimensions):
            if random.random() < self.mutation_rate:
                # 変数に対して小さな変化を加える（変異幅を制限）
                mutation_value = random.uniform(-0.01, 0.01)
                individual[i] += mutation_value
                # 範囲外に出ないように調整
                individual[i] = min(max(individual[i], self.bounds[i][0]), self.bounds[i][1])
        return individual

    def run(self):
        """
        遺伝的アルゴリズムの実行
        """
        population = self.initialize_population()
        best_fitness_history = []
        best_individual = None
        best_fitness = float('inf')

        for gen in range(self.generations):
            selected = self.selection(population)
            next_generation = []
            while len(next_generation) < self.population_size:
                parents = random.sample(selected, 2)
                offspring = self.crossover(parents[0], parents[1])
                next_generation.extend([self.mutate(child) for child in offspring])
            population = next_generation[:self.population_size]
            current_best = min(population, key=lambda x: self.fitness(x))
            current_fitness = self.fitness(current_best)
            best_fitness_history.append(current_fitness)
            if current_fitness < best_fitness:
                best_individual = current_best
                best_fitness = current_fitness
            print(f"Generation {gen}: Best Fitness = {current_fitness}")

        # 最適な個体の詳細を計算
        dv1 = [best_individual[0], best_individual[1], 0]
        dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost = tb.hohman_orbit3(self.x1, self.y1, self.r_aim, dv1)
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'dv1_ans': dv1_ans,
            'dv2_ans': dv2_ans,
            'sol_com': sol_com,
            'sol1': sol1,
            'sol2': sol2,
            'time_cost': time_cost,
            'fitness_history': best_fitness_history
        }
