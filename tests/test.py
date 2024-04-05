import two_body as tb
import orbit_calc as oc

R = 6378.137 # Earth radius
tb.draw_hohman_orbit(*oc.hohmann_pos(R + 200, 35586))