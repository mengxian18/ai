# 引入库
import numpy as np
 
# 定义均方差损失函数
def loss_mse(y_true, y_pred):
    mse_loss = np.mean(np.power(y_true - y_pred, 2))
 
    return mse_loss
    # 构造 y_true 和 y_pred
 #差值平方平均
y_true = np.array([1,2,3,4,5])
y_pred = np.array([1.5, 2, 3, 4.5, 5])
 
# 计算和输出均方差损失函数
mse_loss = loss_mse(y_true, y_pred)
 
print('均方差损失函数：', mse_loss)
 
# 定义交叉熵损失函数
def loss_ce(y_true, y_pred):
 
  # 计算真实标签与预测标签之间的交叉熵
  ce_loss = -np.mean(np.sum(np.multiply(y_true, np.log(y_pred)), axis=1))
  return ce_loss
 
# 构造 y_true 和 y_pred
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.6, 0.2, 0.2], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
 
# 计算和输出交叉熵损失函数
ce_loss = loss_ce(y_true, y_pred) 
print('交叉熵损失函数：',ce_loss)