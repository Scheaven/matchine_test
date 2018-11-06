import numpy as np
import random

def genData(numPoints,bias,variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=(numPoints))

    for i in range(numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = i + bias + random.uniform(0,1)+variance
    return x,y

def gradientDescent(x,y,theta,alpha,m,numIterations):
    xTran = np.transpose(x)
    for i in  range(numIterations):
        yhat = np.dot(x,theta)
        loss = yhat - y
        cost = np.sum(loss**2)/(2*m)
        gradient = np.dot(xTran,loss)/m
        theta-= alpha*gradient
        print("Iteration %d | cost :%f" % (i, cost))
    return  theta


if __name__ == '__main__':
    x, y = genData(100, 25, 10)
    # print(x)
    # print(y)

    m,n = np.shape(x)
    n_y = np.shape(y)

    # print("m:" + str(m) + " n:" + str(n) + " n_y:" + str(n_y))

    numIterations = 100000
    alpha = 0.0005
    theta = np.ones(n)

    print(theta)
    theta = gradientDescent(x, y, theta, alpha, m, numIterations)
    print(theta)
