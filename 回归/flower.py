# 最小二乘法 小花朵数量预测可视化
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simHei']
temperature = [10,15,20,25,30,35]
# temperature=np.linspace[10,15,20,25,30,35]
flower = [136,140,155,160,157,175]
plt.scatter(temperature,flower,c='red',label='小花朵数量随温度的变化量')
plt.plot(temperature,flower,c='blue',linestyle='-')
plt.xlabel('温度')
plt.ylabel('花朵数量')
plt.legend(loc=1)
#plt.show()

# 定义最小二乘法函数
def least_square(X,Y):
    """
    :param X:样本矩阵
    :param Y: 样本标签向量
    :return: 回归系数
    """
    W = (X*X.T).I*X*Y.T # .T表示逆矩阵  .I表示转置
    return W

# 求解回归系数
x_0 = np.ones(6)
x_1 = np.array(temperature)
X = np.mat([x_0,x_1])
Y = np.mat([flower])
W = least_square(X,Y)
W.shape
W[1,0]
W[0,0]
print("线性回归模型为：y={:.2f}x+{:2f}".format(W[1,0],W[0,0]))

# 线性模型可视化
x1 = np.linspace(10,35,100)
y1 = W[1,0]*x1+W[0,0]
plt.plot(x1,y1,c="green",linestyle="--",label="拟合优度")
plt.show()
plt.legend()

# 小花朵数量的预测
temperature_new = np.array([40,5,-3])
flower_new = W[1,0]*temperature_new+W[0,0]
print(flower_new)

# 方法二：调用sklearn
from sklearn.linear_model import LinearRegression
lrg = LinearRegression() # 模型实例化
# lrg.fit(np.mat(temperature).T,np.mat(flower).T) # 模型训练
lrg.fit(np.mat(temperature).reshape(-1,1),np.mat(flower).reshape(-1,1))
lrg.coef_
lrg.intercept_
lrg.predict(np.mat(temperature_new).T)