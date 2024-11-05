import numpy as np
import matplotlib.pyplot as plt

# 1次元の関数
def f(x):
    return x**2

# ガウスノイズを加える
def add_gaussian_noise_1d(x, mu=0, sigma=1, n_samples=1000):
    """
    mu : 平均
    sigma : 標準偏差
    n_samples : ノイズを加えるサンプル数
    """
    noise = np.random.normal(mu, sigma, n_samples)
    return x + noise

# 入力範囲
x = 2  # 例として固定したxの値
noisy_inputs = add_gaussian_noise_1d(x, mu=0, sigma=0.5)  # ノイズを加える

# 出力を計算
outputs = f(noisy_inputs)

# 結果のプロット
plt.hist(outputs, bins=30, alpha=0.7, label='Noisy outputs')
plt.axvline(f(x), color='r', linestyle='--', label='True value (without noise)')
plt.title("Output Distribution with Gaussian Noise")
plt.xlabel("f(x) value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
