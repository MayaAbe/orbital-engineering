import numpy as np
from scipy.integrate import odeint


def odeint(func, y0, t, args=(), rtol=1e-6, atol=1e-12, full_output=False):
    y0 = np.array(y0, dtype=float)
    y = odeint(func, y0, t, args=args, rtol=rtol, atol=atol, full_output=full_output)
    return y


def rk4(func, y0, t, args=(), rtol=1e-6, atol=1e-12, hmax=0.0, full_output=False):
    y0 = np.array(y0, dtype=float)
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    output = {'message': 'Integration successful.'}

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if hmax != 0.0 and dt > hmax:
            dt = hmax

        k1 = np.array(func(y[i - 1], t[i - 1], *args))
        k2 = np.array(func(y[i - 1] + 0.5 * dt * k1, t[i - 1] + 0.5 * dt, *args))
        k3 = np.array(func(y[i - 1] + 0.5 * dt * k2, t[i - 1] + 0.5 * dt, *args))
        k4 = np.array(func(y[i - 1] + dt * k3, t[i - 1] + dt, *args))

        y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Error check based on tolerance settings
        error_estimate = np.max(np.abs(dt / 6 * (k1 + 2*k2 + 2*k3 + k4)))
        if error_estimate > atol + rtol * np.max(np.abs(y[i])):
            output['message'] = 'Integration step failed due to error tolerance.'

    if full_output:
        return y, output
    else:
        return y


def rk7(func, y0, t, args=(), rtol=1e-6, atol=1e-12, hmax=0.0, full_output=False):
    y0 = np.array(y0, dtype=float)
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    output = {'message': 'Integration successful.'}

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if hmax != 0.0 and dt > hmax:
            dt = hmax

        k1 = np.array(func(y[i - 1], t[i - 1], *args))
        k2 = np.array(func(y[i - 1] + (1/5) * dt * k1, t[i - 1] + (1/5) * dt, *args))
        k3 = np.array(func(y[i - 1] + (3/40) * dt * k1 + (9/40) * dt * k2, t[i - 1] + (3/10) * dt, *args))
        k4 = np.array(func(y[i - 1] + (44/45) * dt * k1 - (56/15) * dt * k2 + (32/9) * dt * k3, t[i - 1] + (4/5) * dt, *args))
        k5 = np.array(func(y[i - 1] + (19372/6561) * dt * k1 - (25360/2187) * dt * k2 + (64448/6561) * dt * k3 - (212/729) * dt * k4, t[i - 1] + (8/9) * dt, *args))
        k6 = np.array(func(y[i - 1] + (9017/3168) * dt * k1 - (355/33) * dt * k2 + (46732/5247) * dt * k3 + (49/176) * dt * k4 - (5103/18656) * dt * k5, t[i - 1] + dt, *args))
        k7 = np.array(func(y[i - 1] + (35/384) * dt * k1 + (500/1113) * dt * k3 + (125/192) * dt * k4 - (2187/6784) * dt * k5 + (11/84) * dt * k6, t[i - 1] + dt, *args))

        y[i] = y[i - 1] + (dt / 840) * (35 * k1 + 500 * k3 + 125 * k4 - 2187 * k5 + 11 * k6 + 693 * k7)

        # Error check based on tolerance settings
        error_estimate = np.max(np.abs(dt / 840 * (35 * k1 + 500 * k3 + 125 * k4 - 2187 * k5 + 11 * k6 + 693 * k7)))
        if error_estimate > atol + rtol * np.max(np.abs(y[i])):
            output['message'] = 'Integration step failed due to error tolerance.'

    if full_output:
        return y, output
    else:
        return y
