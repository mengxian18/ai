import numpy as np
import matplotlib.pyplot as plt
# 定义函数：激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 定义函数：梯度下降

def Gradient_Descent(x, y, theta, alpha, iterations):
    for i in range(iterations):
        grad = np.dot(x.T, (sigmoid(np.dot(x, theta)) - y))
        theta = theta - alpha * grad
    return theta
# 生成数据
X = np.random.randn(100, 2) # 生成100个样本，每个样本有2个特征
# 将标签设定为0-1分类
Y = np.array([0 if np.sum(x)<0 else 1 for x in X])
# 添加一列全为1的系数
X = np.c_[np.ones(X.shape[0]), X]
print(X)
# 初始化参数
theta = np.zeros(X.shape[1])
 
# 设定学习率和迭代次数
alpha = 0.01
iterations = 1000
# 调用梯度下降函数
theta = Gradient_Descent(X, Y, theta, alpha, iterations)
# 计算预测值
prediction = sigmoid(np.dot(X, theta))
 
#print(prediction)
 
# 画图查看结果
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(X[:, 1], X[:, 2], c=Y, cmap='viridis')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ax[1].scatter(X[:, 1], X[:, 2], c=prediction, cmap='plasma')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')
plt.show()