from math import cos, sin, rad2deg, deg2rad, sqrt


# 地球の扁平による摂動
def flatPertur():
    return 0


# 月による摂動
def moonPertur():
    n =  # 平均運動
    np = # 月の平均運動
    e =  # 離心率
    ep =  # 月の離心率
    i =  # 軌道傾斜角
    ip =  # 月の軌道傾斜角
    mE =  #地球質量
    mp =  #摂動体質量
    M = mp/(mp + mE)

    dOMEGA = -(3/4)* n * (np /n)**2 * M * cos(rad2deg(i))/sqrt(1 - e**2) * (1+ 3/2 * e**2) * ((1-ep**2)**(-3/2)) * (1-3/2*sin(rad2deg(ip))**2)
    domega = (3/4)* n * (np /n)**2 * M * 1/sqrt(1 - e**2) * (2 - 5/2 *sin(rad2deg(i)) + 0.5*e**2) * ((1-ep**2)**(-3/2)) *  (1-3/2*sin(rad2deg(ip))**2)
    dM = n - n/4 * (np/n)**2 * M * (7+3*e**2)* (1 - 3/2 * sin(rad2deg(i))**2) * ((1-ep**2)**(-3/2)) * (1-3/2*sin(rad2deg(ip))**2)

    return dOMEGA, domega, dM
