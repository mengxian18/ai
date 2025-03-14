import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import SimpleRNN, Dense
 
# 设置超参数
timesteps = 10
input_dim = 2
output_dim = 3
 
# 定义输入层
inputs = Input(shape=(timesteps, input_dim))
 
# 定义隐藏层
x = SimpleRNN(units=output_dim)(inputs)
 
# 定义输出层
predictions = Dense(units=output_dim, activation='softmax')(x)
 
# 构建模型
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
 
# 模拟输入数据
x_train = np.random.random((1000, timesteps, input_dim))
y_train = np.random.random((1000, output_dim))
 
# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32)