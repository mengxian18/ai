import numpy as np
 
# 定义sigmoid函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))
 
# 定义损失函数
def cost(x, y, w):
    return np.sum(np.power((sigmoid(np.dot(x, w)) - y), 2))
 
# 定义一元一次函数感知器
def linear_perceptron(x, y, w, learning_rate, epochs):
    for i in range(epochs):
        z = np.dot(x, w)
        a = sigmoid(z)
        e = a - y
        w = w - learning_rate * np.dot(x.T, e)#梯度下降算法更新权重
        cost_value = cost(x, y, w)
        print("Epoch %d, cost %f" % (i, cost_value))
    return w
 
# 设置超参数
learning_rate = 0.01
epochs = 500
 
# 构建输入输出数据
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
 
# 初始化权重
w = np.array([0.1, 0.1])
 
# 训练感知器
w = linear_perceptron(x, y, w, learning_rate, epochs)
 
# 打印训练结果
print("Final weight:", w)