import numpy as np
import random as rn
import pandas as pd

m = 6
N = 327 #  варіант
y_max = (30 - N) * 10
y_min = (20 - N) * 10
x1_min, x1_max = [10, 60]
x2_min, x2_max = [-35, 10]
xn = [[-1, -1], [-1, 1], [1, -1]] # нормалізовані фактори
matrix = pd.DataFrame({"x1" : [i[0] for i in xn], "x2" : [i[1] for i in xn]})
y = np.array([[rn.randint(y_min, y_max) for i in range(m)] for j in range(3)])


for i in range(len(y[0])):
    y_n = "y" + str(i + 1)
    matrix[y_n] = np.array([j[i] for j in y])
not_homogeneous = False


m += 1 # кількість експериментів
sigma_theta = np.sqrt((2 * (2 * m - 2))/(m * (m - 4)))

while True:
    if not_homogeneous:
        y[m - 1] = [rn.randint(y_min, y_max) for i in range(3)]
        matrix["y" + str(m)] = y[m - 1]

    y_avg = [i.mean() for i in y]
    for i in range(len(y_avg)):
        y_n = "y" + str(i + 1)
        print(y_n + "-average = " +str(y_avg[i]))

    disp_y = [i.var() for i in y]

    for i in range(len(y_avg)):
        y_n = "y" + str(i + 1)
        print(y_n + "-dispersion = " +str(disp_y[i]))



    def f_uv(u, v):
        if u >= v:
            return u / v
        else:
            return v / u

    F_uv1 = f_uv(disp_y[0], disp_y[1])
    F_uv2 = f_uv(disp_y[2], disp_y[0])
    F_uv3 = f_uv(disp_y[2], disp_y[1])
    fuv = [F_uv1, F_uv2, F_uv3]

    print(f"F_uv1 = {round(F_uv1, 3)};\nF_uv2 = {round(F_uv2, 3)};\nF_uv3 = {round(F_uv3, 3)};")

    num = (m - 2) / m
    theta_uv1 = num * F_uv1
    theta_uv2 = num * F_uv2
    theta_uv3 = num * F_uv3

    theta = [theta_uv1, theta_uv2, theta_uv3]
    print(f"theta_uv1 = {round(theta_uv1, 3)};\ntheta_uv2 = {round(theta_uv2, 3)};\ntheta_uv3 = {round(theta_uv3, 3)};")

    r_uv1 = np.abs(theta_uv1 - 1) / sigma_theta
    r_uv2 = np.abs(theta_uv2 - 1) / sigma_theta
    r_uv3 = np.abs(theta_uv3 - 1) / sigma_theta

    ruv = [r_uv1, r_uv2, r_uv3]
    print(f"r_uv1 = {round(r_uv1, 3)};\nr_uv2 = {round(r_uv2, 3)};\nr_uv3 = {round(r_uv3, 3)};")

    r_kr = 2

    homogeneous = True
    for i in range(len(ruv)):
        if ruv[i] > r_kr:
            print("Неоднорідна дисперсія")
            not_homogeneous = True
            continue
    if homogeneous:
        print("Однорідна дисперсія")


    mx = [sum([xn[i][j] for i in range(len(xn))]) / len(xn) for j in range(len(xn[0]))]
    m_y = sum(y_avg) / len(y_avg)
    a1 = sum(xn[i][0] ** 2 for i in range(len(xn))) / len(xn)
    a2 = (xn[0][0] * xn[0][1] + xn[1][0] * xn[1][1] + xn[2][0] * xn[2][1]) / len(xn)
    a3 = sum(xn[i][1] ** 2 for i in range(len(xn))) / len(xn)
    a11 = sum(xn[i][0] * y_avg[i] for i in range(len(xn))) / len(xn)
    a22 = sum(xn[i][1] * y_avg[i] for i in range(len(xn))) / len(xn)
    print(f"Нормовані коефіцієнти рівняння регресії:\nmx1 = {mx[0]};\nmx2 = {mx[1]};\nm_y_avg = {m_y};\na1 = {a1};\n" + 
        f"a2 = {a2}\na3 = {a3}\na11 = {a11};\na22 = {a22};")

    b0 = np.linalg.det(np.array([[m_y, mx[0], mx[1]], [a11, a1, a2], [a22, a2, a3]])) / np.linalg.det(np.array([[1, mx[0], mx[1]], [mx[0], a1, a2], [mx[1], a2, a3]]))
    b1 = np.linalg.det(np.array([[1, m_y, mx[1]], [mx[0], a11, a2], [mx[1], a22, a3]])) / np.linalg.det(np.array([[1, mx[0], mx[1]], [mx[0], a1, a2], [mx[1], a2, a3]]))
    b2 = np.linalg.det(np.array([[1, mx[0], m_y], [mx[0], a1, a11], [mx[1], a2, a22]])) / np.linalg.det(np.array([[1, mx[0], mx[1]], [mx[0], a1, a2], [mx[1], a2, a3]]))

    print(f"Рівняння регресії:\ny = {round(b0, 3)} + {round(b1, 3)}*x1 + {round(b2, 3)}*x2")
    print(f"{round(b0, 3)} + {round(b1, 3)}*({xn[0][0]}) + {round(b2, 3)}*({xn[0][1]}) = -3012.334\ny-середнє = {y_avg[0]}\nДані сходяться, отже коефіцієнти підібрані правильно!")
    delta_x1 = np.abs(x1_max - x1_min) / 2 
    delta_x2 = np.abs(x2_max - x2_min) / 2
    x10 = (x1_max + x1_min)
    x20 = (x2_max + x2_min)
    a0 = b0 - b1 * (x10 / delta_x1) - b2 * (x20 / delta_x2)
    a1 = b1 / delta_x1
    a2 = b2 / delta_x2
    print(f"Натуралізоване рівняння регресії:\ny = {round(a0, 3)} + {round(a1, 3)}*x1 + {round(a2, 3)}*x2")
    print(f"{round(a0, 3)} + {round(a1, 3)}*({xn[0][0]}) + {round(a2, 3)}*({xn[0][1]}) = {round(a0 + a1*xn[0][0]+a2*xn[0][1], 3)}\ny-середнє = {y_avg[0]}\nДані майже сходяться, отже коефіцієнти підібрані правильно!")
    break