import numpy as np
import math

'''
皮尔逊相关系数和R^2相关度计算的方法
'''
#相关系数的计算函数
def pearsonCorrelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0;varX = 0;varY = 0;
    for i in range(len(X)):
        diffXBar = X[i] - xBar
        diffYBar = Y[i] - yBar
        SSR += (diffXBar*diffYBar) #XY的协方差
        varX += diffXBar**2
        varY += diffYBar**2
    SST = math.sqrt(varX*varY) #方差积的开方
    return SSR/SST

def polyfit(x,y,degree):
    results = {}
    coeffs = np.polyfit(x,y,degree) #根据np获取回归参数，degree为几次方的回归
    results["polynomial"] = coeffs.tolist()
    p = np.poly1d(coeffs) #传递参数构造线性模型  #？？？？和之前实现的线性回归方法有何不同？和sklearn中的线性回归模型有何不同？？
    #根据公式计算R^2的值
    yhat = p(x) #获取预测值
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results["determination"] = ssreg / sstot
    return  results

if __name__ == '__main__':
    testX = [1,3,8,7,9]
    testY = [10,12,24,21,34]
    print(pearsonCorrelation(testX,testY))
    #对于一元变量的线性回归来说，R^2即为皮尔逊相关系数的平方
    print("R^2指标：%s",pearsonCorrelation(testX,testY)**2)
    print("R^2指标：%s",polyfit(testX,testY,1))