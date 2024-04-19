import numpy as np
import matplotlib.pyplot as plt

def model(y, t):
    GM = 398600.4354360959  # 地球の重力定数, km^3/s^2
    r = np.linalg.norm(y[:3])
    dxdt = [
        y[3],
        y[4],
        y[5],
        -GM * y[0] / (r**3),
        -GM * y[1] / (r**3),
        -GM * y[2] / (r**3)
    ]
    return np.array(dxdt)

def runge_kutta_odeint(func, y0, t, args=(), rtol=1e-6, atol=1e-12, hmax=0.0, full_output=False):
    y0 = np.array(y0, dtype=float)
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    output = {'message': 'Integration successful.'}

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if hmax != 0.0 and dt > hmax:
            dt = hmax

        k1 = func(y[i - 1], t[i - 1], *args)
        k2 = func(y[i - 1] + 0.5 * dt * k1, t[i - 1] + 0.5 * dt, *args)
        k3 = func(y[i - 1] + 0.5 * dt * k2, t[i - 1] + 0.5 * dt, *args)
        k4 = func(y[i - 1] + dt * k3, t[i - 1] + dt, *args)

        y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Error check based on tolerance settings
        error_estimate = np.max(np.abs(dt / 6 * (k1 + 2*k2 + 2*k3 + k4)))
        if error_estimate > atol + rtol * np.max(np.abs(y[i])):
            output['message'] = 'Integration step failed due to error tolerance.'

    if full_output:
        return y, output
    else:
        return y

# Initial conditions for the two-body problem
y0 = [6400, 0, 0, 0, 7.8, 0]  # Earth radius + altitude, position (x, y, z) and velocity (vx, vy, vz)

# Time array
t = np.linspace(0, 86400, 1000)  # Time span for one day, with 1000 time points

# Solve the ODE
result = runge_kutta_odeint(model, y0, t)

# Plotting the result
plt.plot(result[:, 0], result[:, 1])
plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Orbit Plot using Runge-Kutta Method')
plt.grid(True)
plt.axis('equal')
plt.show()
