from numpy import genfromtxt
from sklearn import linear_model

dataPath = r"Delivery.csv"
deliverData = genfromtxt(dataPath,delimiter=",") #参数分别为数据路径和数据间的分隔符
# print(deliverData)

x = deliverData[:,:-1]
y = deliverData[:,-1]
# print(x,y)


lr = linear_model.LinearRegression() #获取线性分类模型
lr.fit(x,y)

print (lr)

print(lr.coef_) #所有权重系数
print(lr.intercept_) #除了偏转的系数

xPredict = [102,6]
yPredict = lr.predict(xPredict)
print(yPredict)

