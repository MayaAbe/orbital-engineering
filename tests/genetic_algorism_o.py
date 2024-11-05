def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    start_time = time.time()
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    end_time = time.time()

    print(f'所要時間: {end_time - start_time} 秒')

    best_ind = hof[0]
    print(f'最良個体: {best_ind}')
    print(f'最良適応度: {best_ind.fitness.values[0]}')

    # 最良結果のパラメータを抽出
    dv1_x, dv1_y = best_ind
    dv1 = [dv1_x, dv1_y, 0]
    dv1_ans, dv2_ans, sol_com, sol1, sol2, time_cost = tb.hohman_orbit3([384400+3000, 0, 0, 0, 1.022+1.02, 0], [384400, 0, 0, 0, 1.022, 0], 6378.137 + 35786, dv1)

    # プロット
    plt.plot(sol1[:, 0], sol1[:, 1], 'b', label='dv1 前')
    plt.plot(sol_com[:, 0], sol_com[:, 1], 'k', label='軌道')
    plt.plot(sol2[:, 0], sol2[:, 1], 'r--', label='目標軌道')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

    energy = np.array([oc.energy(sol_com[i]) for i in range(len(sol_com))])
    plt.figure()
    plt.plot(energy)
    plt.xlabel('時間')
    plt.ylabel('エネルギー')
    plt.title('エネルギー変動')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
