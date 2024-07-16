import orbit_calc as oc

hight = [200, 500, 1000, 10000, 36000]
R = 6378.137 # Earth radius

for i in range(len(hight)):
    print(f'v of {hight[i]}km is {oc.v_circular([R+hight[i], 0, 0])}')
    print(f'T1 of {hight[i]}km is {oc.T_circular([R+hight[i], 0, 0])}')
    print(f'T2 of {hight[i]}km is {oc.T_owbow([R+hight[i], 0, 0, 0, oc.v_circular([R+hight[i], 0, 0]), 0])}')
    print("-----------------")