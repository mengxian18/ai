# 构造数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# 模型参数
w = 1.0  # 初始化w
# 模型定义
def forward(x):
    return x * w
# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
# 梯度
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)
# 更新参数
def update():
    global w
    w = w - 0.01 * gradient(x_data[0], y_data[0])
# 打印结果
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))