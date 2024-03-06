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
plt.show()