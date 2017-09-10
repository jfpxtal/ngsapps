from math import nan

euler = [[0],
         [nan, 1]]

rk4 = [[0],
       [1/2, 1/2],
       [1/2, 0, 1/2],
       [1, 0, 0, 1],
       [nan, 1/6, 1/3, 1/3, 1/6]]

def RungeKutta(but, h, f, t, y):
    ks = []
    res = y.CreateVector()
    res.data = y
    for k in range(len(but)-1):
        ynew = y.CreateVector()
        ynew.data = y
        for l in range(k):
            ynew.data += h*but[k][l+1]*ks[l]
        knew = f(t+h*but[k][0], ynew)
        res.data += h*but[-1][k+1]*knew
        ks.append(knew)
    return res
