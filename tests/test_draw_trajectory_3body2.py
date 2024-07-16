import core.two_body as tb
import core.orbit_calc as oc
import numpy as np

# 長さの単位はkm, 時間の単位はs

R = 6378.137 # Earth radius
x1 = [384400+3000, 0, 0, 0, 1.022+1.02, 0]  # 3000km
y1 = [384400, 0, 0, 0, 1.022, 0]
dv1 = [0.84, -2.68, 0] # -2.898979

# print(oc.T_owbow(x1))
dv1, dv2, c, d, e, f = tb.draw_hohman_orbit3(x1, y1, R + 35786, dv1)
print(f"dv1: {dv1}, dv2: {dv2}")
