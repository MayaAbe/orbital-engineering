import orbit_calc as oc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

R = 6378.137  # Earth radius


def func(x, t):
    GM = 398600.4354360959  # 地球の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3),
            -GM*x[1]/(r**3),
            -GM*x[2]/(r**3)]
    return dxdt


def funcMoon(y, t):
    GM = 403493.253  # 月の重力定数, km3/s2
    r = np.linalg.norm(y[0:3])
    dydt = [y[3],
            y[4],
            y[5],
            -GM*y[0]/(r**3),
            -GM*y[1]/(r**3),
            -GM*y[2]/(r**3)]
    return dydt


def funcTri(x, y, t):
    GM = 398600.4354360959  # 地球の重力定数, km3/s2
    GMm = 4904.058  # 月の重力定数, km3/s2
    r = np.linalg.norm(x[0:3])
    r_m = np.linalg.norm(y[0:3])
    z = [x[0]-y[0], x[1]-y[1], x[2]-y[2]]
    r_z = np.linalg.norm(z)
    dxdt = [x[3],
            x[4],
            x[5],
            -GM*x[0]/(r**3) - (GMm*z[0]/(r_z**3) + GMm*y[0]/(r_m**3)),
            -GM*x[1]/(r**3) - (GMm*z[1]/(r_z**3) + GMm*y[1]/(r_m**3)),
            -GM*x[2]/(r**3) - (GMm*z[2]/(r_z**3) + GMm*y[2]/(r_m**3))]
    return dxdt


x1 = [R+36000, 0, 0, 0, 3.0668882673431845, 0]
y1 = [384400, 0, 0, 0, 1.022, 0]

Tx = oc.T_circular(x1)  # 公転周期
Ty = oc.T_circular(y1)  # 公転周期
t0 = np.linspace(0, Tx, 1000)
t1 = np.linspace(0, Ty, 1000)

print(Tx)
print(oc.T_owbow(x1))

"""solx = odeint(func, x1, t0)
soly = odeint(funcMoon, y1, t1)


plt.plot(solx[:, 0], solx[:, 1], 'b')
plt.plot(soly[:, 0], soly[:, 1], 'r')
plt.grid()
plt.gca().set_aspect('equal')
plt.show()"""