import core.two_body as tb
import core.orbit_calc as oc

# 長さの単位はkm, 時間の単位はs

R = 6378.137 # Earth radius
x1 = [R+200, 0, 0, 0, 7.784261686425335, 0]
dv1 = [0, 2.5,0]

# print(oc.T_owbow(x1))
tb.draw_hohman_orbit3(x1, R + 35786, dv1)
