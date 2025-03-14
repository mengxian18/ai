import torch
# 定义一个简单的线性模型
model = torch.nn.Linear(1, 1)
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 定义损失函数pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
criterion = torch.nn.MSELoss()
# 训练数据
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
# 训练模型
for epoch in range(1000):
    # 计算模型输出
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    # 清空梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 打印训练结果
    if (epoch+1) % 20 == 0:
        print(f'epoch {epoch+1}: w = {model.weight.item():.3f}, loss = {loss.item():.8f}')
 
# 训练完成
w = model.weight.item()
print(f'Predict after training: f(5) = {5*w:.3f}')
 
#模型保存
torch.save(model.state_dict(), 'model.pth')
 
#模型加载
state_dict  = torch.load('model.pth')
model.load_state_dict(state_dict )
 
#模型预测
predict = model(torch.tensor([5.0]))
print(predict.detach().numpy()[0])