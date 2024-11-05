# functions.py

import numpy as np

def sphere_function(x):
    """
    Sphere関数：f(x) = Σ(x_i)^2
    最小値は0で、全てのx_iが0のときに達成される。
    """
    return sum([xi**2 for xi in x])

def rastrigin_function(x):
    """
    Rastrigin関数：多峰性の関数で、局所解が多い。
    最小値は0で、全てのx_iが0のときに達成される。
    """
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
