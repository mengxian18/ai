import numpy as np
from tensorflow.keras.models import Sequential  # 导入keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import tushare as ts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
 
# 生成相应的数据函数
def get_beans4(counts):
    xs = np.random.rand(counts, 2) * 2
    ys = np.zeros(counts)
    for i in range(counts):
        x = xs[i]
        if (np.power(x[0] - 1, 2) + np.power(x[1] - 0.3, 2)) < 0.5:
            ys[i] = 1
 
    return xs, ys
 
 
# 画出数据的散点图
def show_scatter(X, Y):
    if X.ndim > 1:
        show_3d_scatter(X, Y)
 
    else:
        plt.scatter(X, Y)
        plt.show()
 
 
# 画3d散点图
def show_3d_scatter(X, Y):
    x = X[:, 0]
 
 
    z = X[:, 1]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, z, Y)
    plt.show()
 
 
# 画3D图
def show_scatter_surface_with_model(X, Y, model):
    # model.predict(X)
    x = X[:, 0]
    z = X[:, 1]
    y = Y
 
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, z, y)
 
    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)
 
    X = np.column_stack((x[0], z[0]))
 
    for j in range(z.shape[0]):
        if j == 0:
            continue
        X = np.vstack((X, np.column_stack((x[0], z[j]))))
 
    y = model.predict(X)
 
    # return
    # y = model.predcit(X)
    y = np.array([y])
    y = y.reshape(x.shape[0], z.shape[1])
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()

m = 100 # 数据量
X, Y = get_beans4(m)
show_scatter(X, Y)
print(X)
print(X.shape)
 
model = Sequential()
model.add(Dense(units=10, activation='sigmoid', input_dim=2))
# units 神经元个数， activation激活函数类型， 输了特征维度
model.add(Dense(units=1, activation='sigmoid')) # 输出层
# 编译网络
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.3), metrics=['accuracy'])
# mean_squared_error 均方误差 sgd 随机梯度下降算法 accuracy 准确度
 
# 训练回合数epochs， batch_size 批数量，一次训练利用多少样本
model.fit(X, Y, epochs=8000, batch_size=64)

# 预测函数
pres = model.predict(X)
show_scatter_surface_with_model(X, Y, model) # 三维的