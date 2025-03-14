def perceptron(x):
    w = 0.5  # 初始化权重
    b = 0.1  # 初始化偏置
    y_hat = w * x + b  # 计算预测值
    if y_hat >= 0.0:
        return 1
    else:
        return 0
 
 
# 调用perceptron()函数
prediction = perceptron(5)
 
print('预测值为：', prediction)
import numpy as np
# X 代表输入, Y 代表输出
X = np.array([[1,2], [3,4], [5,6]])
Y = np.array([2,4,6])
 
# 定义权重向量
weights = np.array([0.5, 0.5])
 
print("权重向量:", weights)
# 定义偏置项
bias = 1
 
# 权重与偏置项输出函数
outputs = np.dot(X, weights) + bias

print("输出函数:", outputs)
# 计算损失函数
error = Y - outputs
 
# 更新权重向量参数和偏置项参数
weights = weights + 0.1 * np.dot(X.T, error)
bias = bias + 0.1 * np.sum(error)
 
# print the results
print("Updated weights:", weights)
print("Updated bias:", bias)