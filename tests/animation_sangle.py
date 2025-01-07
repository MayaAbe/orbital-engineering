import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation, FFMpegWriter, PillowWriter
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# 二体問題の運動方程式（地球-月系）
def funcMoon(x, t):
    GM = 403493.253  # 月の重力定数, km^3/s^2
    r = np.linalg.norm(x[0:3])
    dxdt = [
        x[3],
        x[4],
        x[5],
        -GM * x[0] / (r**3),
        -GM * x[1] / (r**3),
        -GM * x[2] / (r**3),
    ]
    return dxdt

def funcTri(x, t, y_interpolated):
    y = y_interpolated(t)
    GM = 398600.4354360959  # 地球の重力定数, km^3/s^2
    GMm = 4904.058  # 月の重力定数, km^3/s^2
    r = np.linalg.norm(x[0:3])
    r_m = np.linalg.norm(y[0:3])
    z = [x[0] - y[0], x[1] - y[1], x[2] - y[2]]
    r_z = np.linalg.norm(z)
    dxdt = [
        x[3],
        x[4],
        x[5],
        -GM * x[0] / (r**3)
        - (GMm * z[0] / (r_z**3) + GMm * y[0] / (r_m**3)),
        -GM * x[1] / (r**3)
        - (GMm * z[1] / (r_z**3) + GMm * y[1] / (r_m**3)),
        -GM * x[2] / (r**3)
        - (GMm * z[2] / (r_z**3) + GMm * y[2] / (r_m**3)),
    ]
    return dxdt

def MoonEarthSat(x: tuple, y: tuple, n: int, step: int):
    # 衛星と月の軌道を計算する関数
    Tx = 86164  # 地球の自転周期（秒）
    Ty = 2360591.744  # 月公転周期（秒）
    m = int(np.ceil(n * Tx / Ty))
    tx = np.linspace(0, n * Tx, int(n * Tx / step))
    ty = np.linspace(0, m * Ty, int(m * Ty / step))
    soly = odeint(funcMoon, y, ty)
    y_interpolated = interp1d(
        ty, soly, axis=0, kind="cubic", fill_value="extrapolate"
    )
    solx = odeint(lambda x, t: funcTri(x, t, y_interpolated), x, tx)
    solx = np.asarray(solx)
    soly = np.asarray(soly)
    return solx, soly

def create_orbit_animation(
    x1,
    y1,
    start_fraction=0.0,
    end_fraction=1.0,
    step=2000,
    interval=10,
    highlight_latest=True,
    speed_factor=0.5,
    save=False,
    filename="orbit_animation.mp4",
    Omega=np.pi / 60,  # 円錐の立体角を指定
):
    solx, soly = MoonEarthSat(x1, y1, 6, 10)

    # 表示するフレームの範囲を設定
    start_index = int(len(solx) * start_fraction)
    end_index = int(len(solx) * end_fraction)
    solx = solx[start_index:end_index:step]
    soly = soly[start_index:end_index:step]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 軌道の範囲を設定
    ax.set_xlim([np.min(solx[:, 0]), np.max(solx[:, 0])])
    ax.set_ylim([np.min(solx[:, 1]), np.max(solx[:, 1])])
    ax.set_zlim([np.min(solx[:, 2]), np.max(solx[:, 2])])

    # 軸のアスペクト比を同じに設定
    def set_aspect_equal(ax):
        extents = np.array(
            [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
        )
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        ax.set_xlim3d(centers[0] - r, centers[0] + r)
        ax.set_ylim3d(centers[1] - r, centers[1] + r)
        ax.set_zlim3d(centers[2] - r, centers[2] + r)

    set_aspect_equal(ax)

    # 地球を半径6371 kmの球体として描画
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_x = 6371 * np.cos(u) * np.sin(v)
    earth_y = 6371 * np.sin(u) * np.sin(v)
    earth_z = 6371 * np.cos(v)
    earth_surf = ax.plot_surface(earth_x, earth_y, earth_z, color='b', alpha=0.6)

    frames = []
    for i in range(len(solx)):
        artists = []

        # 軌道を描画
        line_solx, = ax.plot(solx[:i+1, 0], solx[:i+1, 1], solx[:i+1, 2], 'b')
        line_soly, = ax.plot(soly[:i+1, 0], soly[:i+1, 1], soly[:i+1, 2], 'r')
        artists.extend([line_solx, line_soly])

        # 最新の位置を強調表示
        if highlight_latest:
            current_solx, = ax.plot([solx[i, 0]], [solx[i, 1]], [solx[i, 2]], 'bo', markersize=8)
            current_soly, = ax.plot([soly[i, 0]], [soly[i, 1]], [soly[i, 2]], 'ro', markersize=8)
            artists.extend([current_solx, current_soly])

        # 円錐の描画
        # 宇宙機と地球の位置
        spacecraft_pos = solx[i, 0:3]
        earth_pos = np.array([0, 0, 0])

        # 開き角 θ の計算
        theta = np.arccos(1 - Omega / (2 * np.pi))

        # 円錐の高さを地球-宇宙機の距離と等しく設定
        cone_height = np.linalg.norm(spacecraft_pos - earth_pos)
        r_base = cone_height * np.tan(theta)

        # 円錐のメッシュを生成
        num_cone_points = 20  # 円錐の滑らかさ
        z = np.linspace(0, -cone_height, 10)  # 高さ方向
        theta_grid = np.linspace(0, 2 * np.pi, num_cone_points)
        Z, Theta_grid = np.meshgrid(z, theta_grid)
        R = (-Z) * np.tan(theta)

        Xc = R * np.cos(Theta_grid)
        Yc = R * np.sin(Theta_grid)
        Zc = Z

        # ローカル座標から宇宙機-地球方向への変換
        direction = (spacecraft_pos - earth_pos) / cone_height

        # 基底ベクトルの作成
        up = np.array([0, 0, 1])
        if np.allclose(direction, up) or np.allclose(direction, -up):
            up = np.array([1, 0, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)

        # 回転行列の作成
        R_matrix = np.stack([right, up, direction], axis=1)

        # メッシュを1次元配列に変換
        Xc_flat = Xc.flatten()
        Yc_flat = Yc.flatten()
        Zc_flat = Zc.flatten()

        # 円錐の表面を計算
        cone_points = np.dot(R_matrix, np.vstack([Xc_flat, Yc_flat, Zc_flat]))
        cone_points += spacecraft_pos[:, np.newaxis]

        # 再度メッシュ形状に戻す
        Xc_plot = cone_points[0].reshape(Xc.shape)
        Yc_plot = cone_points[1].reshape(Xc.shape)
        Zc_plot = cone_points[2].reshape(Xc.shape)

        # 円錐をプロット（透明度を90%に設定：alpha=0.1）
        surf = ax.plot_surface(
            Xc_plot,
            Yc_plot,
            Zc_plot,
            color="g",
            alpha=0.1,
            linewidth=0,
            antialiased=False
        )
        artists.append(surf)

        # 地球の球体を追加
        artists.append(earth_surf)

        frames.append(artists)

    ani = ArtistAnimation(fig, frames, interval=interval * speed_factor * 1000, blit=False)

    if save:
        if filename.endswith(".mp4"):
            Writer = FFMpegWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
            ani.save(filename, writer=Writer)
        elif filename.endswith(".gif"):
            ani.save(filename, writer="pillow")
        else:
            raise ValueError("Unsupported file format. Please use .mp4 or .gif")

    plt.legend()
    plt.grid()
    plt.show()

# 使用例
R = 6371  # 地球の半径, km
x1 = [384400 + 3000, 0, 0, 0.8195, 1.022 + 1.02 - 2.6560, 0]
y1 = [384400, 0, 0, 0, 1.022, 0]

# アニメーションを表示
create_orbit_animation(
    x1,
    y1,
    start_fraction=0.0,
    end_fraction=1.0,
    step=2000,
    highlight_latest=True,
    speed_factor=0.5,
    save=False,
    Omega=np.pi / 60,
)
