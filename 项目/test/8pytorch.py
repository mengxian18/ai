import numpy as np
 
np.random.seed(0)
NUM_SAMPLES = 1000
NUM_FEATURES = 20
 
X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)
y = np.random.randint(0, 2, (NUM_SAMPLES,))
 
print("X shape:", X.shape)
print("y shape:", y.shape)
 
 
### 构建简单的神经网络
 
#接下来，我们使用PyTorch框架来构建一个简单的多层感知机。这个感知机包括一个输入层、一个隐藏层和一个输出层。
 
import torch
import torch.nn as nn
 
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
 
INPUT_SIZE = 20
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
 
model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
 
print(model)
 
 
### 训练神经网络
 
#现在，我们将训练这个简单神经网络。为了使用GPU进行训练，请确保已经安装了适当的PyTorch GPU版本。
 
# 判断是否有GPU可用，如果有，则将模型和数据移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
 
 
### 训练循环
 
#下面的代码将执行训练循环，并在每个循环后输出训练损失。
 
# 超参数设置
learning_rate = 0.001
num_epochs = 500
batch_size = 40
num_batches = NUM_SAMPLES // batch_size
 
# 创建优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
 
# 转换数据为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
 
# 训练循环
for epoch in range(num_epochs):
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        inputs = X[start:end].to(device)  # 将数据移动到GPU
        targets = y[start:end].to(device)  # 将数据移动到GPU
 
        outputs = model(inputs)
        loss = criterion(outputs, targets)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")