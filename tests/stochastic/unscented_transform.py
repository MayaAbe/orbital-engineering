import numpy as np

# シグマポイント生成関数
def generate_sigma_points(mu, cov, alpha=1e-3, kappa=2, beta=0):
    """
    mu: 平均ベクトル
    cov: 共分散行列
    alpha: スケーリングパラメータ
    kappa: スケーリングパラメータ
    beta: スケーリングパラメータ
    """
    n = len(mu)
    lambda_ = alpha**2 * (n + kappa) - n

    # シグマポイントの初期化
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = mu

    # 共分散行列の平方根を計算
    sqrt_matrix = np.linalg.cholesky((n + lambda_) * cov)

    for i in range(n):
        sigma_points[i + 1] = mu + sqrt_matrix[i]
        sigma_points[n + i + 1] = mu - sqrt_matrix[i]

    return sigma_points, lambda_

# Unscented Transform
def unscented_transform(mu, cov, f, alpha=1e-3, kappa=2, beta=0):
    """
    mu: 平均ベクトル (入力)
    cov: 共分散行列 (入力)
    f: 適用する非線形関数
    alpha, kappa, beta: スケーリングパラメータ
    """
    n = len(mu)

    # シグマポイント生成
    sigma_points, lambda_ = generate_sigma_points(mu, cov, alpha, kappa, beta)

    # 重みの計算
    Wm = np.zeros(2 * n + 1)
    Wc = np.zeros(2 * n + 1)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2 * n + 1):
        Wm[i] = Wc[i] = 1 / (2 * (n + lambda_))
        print(f"Wm{i} -> {Wm[i]}")

    # シグマポイントに関数 f を適用
    y = np.zeros(2 * n + 1)
    for i, sigma_point in enumerate(sigma_points):
        y[i] = f(sigma_point)
        print(f"y{i} -> {y[i]}")

    # 出力の平均
    mean_y = np.sum(Wm * y)

    # 出力の分散
    diff_y = y - mean_y
    cov_y = np.sum(Wc * (diff_y**2))

    return mean_y, cov_y

# 非線形関数 f1 (例えば、2変数の二乗和関数)
def f1(x):
    return x[0]**2 + x[1]**2

# 非線形関数 f2 (例えば、2変数の積)
def f2(x):
    return x[0] * x[1]


# 使用例
if __name__ == "__main__":
    # 入力の平均と共分散
    mu = np.array([2, 3])  # 2次元の平均ベクトル
    cov = np.array([[1, 0], [0, 1]])  # 2次元の共分散行列

    # f1 による Unscented Transform
    mean_y1, cov_y1 = unscented_transform(mu, cov, f1)
    print(f"f1: 平均 = {mean_y1}, 分散 = {cov_y1}")

    # f2 による Unscented Transform
    mean_y2, cov_y2 = unscented_transform(mu, cov, f2)
    print(f"f2: 平均 = {mean_y2}, 分散 = {cov_y2}")
