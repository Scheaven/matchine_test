import numpy as np
import random
def getData(numPoint,bias,variance):
    x = np.zeros(shape=(numPoint,2))
    y = np.zeros(shape=(numPoint,1))
    for i in range(numPoint):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i+bias) + random.uniform(0,1) + variance
    return x,y

#
# def gradientDescent(x, y, theta, alpha, m, numIterations):
#     '''dot函数是np中的矩阵乘法，
#         x.dot(y) 等价于 np.dot(x,y)
#         x是m*n 矩阵 ，y是n*m矩阵
#         则x.dot(y) 得到m*m矩阵'''


if __name__ == '__main__':
    x,y = getData(100,25,10)
    # print('x:%s,y:%s',x,y)

    m,n = np.shape(x)
    n_y = np.shape(y)

    print("m:" + str(m) + " n:" + str(n) + " n_y:" + str(n_y))


