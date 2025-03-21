# 1-基于SqueezeNet的眼疾识别

## 背景

#### 1.1 数据集介绍

`iChallenge-PM`是百度大脑和中山大学中山眼科中心联合举办的iChallenge比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。

- training.zip：包含训练中的图片和标签
- validation.zip：包含验证集的图片
- valid_gt.zip：包含验证集的标签

该数据集是从AI Studio平台中下载的，具体信息如下：

https://pan.baidu.com/s/1yBkVHZw5YFYUdqD2GntRyQ?pwd=2357

![9bf63e5bba326a80175de4736304b2a0](D:/learn/笔记/人工智能/人工智能笔记/image/9bf63e5bba326a80175de4736304b2a0.png)

#### 1.2 数据集文件结构

数据集中共有三个压缩文件，分别是：

![Snipaste_2025-03-14_17-31-11](D:/learn/笔记/人工智能/人工智能笔记/image/Snipaste_2025-03-14_17-31-11.png)

#### 2.1 数据标签划分

该眼疾数据集格式有点复杂，这里我对数据集进行了自己的处理，将训练集和验证集写入txt文本里面，分别对应它的图片路径和标签。

```python
import os
import pandas as pd
# 将训练集划分标签
train_dataset = r"F:\SqueezeNet\data\PALM-Training400\PALM-Training400"
train_list = []
label_list = []


train_filenames = os.listdir(train_dataset)

for name in train_filenames:
    filepath = os.path.join(train_dataset, name)
    train_list.append(filepath)
    if name[0] == 'N' or name[0] == 'H':
        label = 0
        label_list.append(label)
    elif name[0] == 'P':
        label = 1
        label_list.append(label)
    else:
        raise('Error dataset!')


with open('F:/SqueezeNet/train.txt', 'w', encoding='UTF-8') as f:
    i = 0
    for train_img in train_list:
        f.write(str(train_img) + ' ' +str(label_list[i]))
        i += 1
        f.write('\n')
# 将验证集划分标签
valid_dataset = r"F:\SqueezeNet\data\PALM-Validation400"
valid_filenames = os.listdir(valid_dataset)
valid_label = r"F:\SqueezeNet\data\PALM-Validation-GT\PM_Label_and_Fovea_Location.xlsx"
data = pd.read_excel(valid_label)
valid_data = data[['imgName', 'Label']].values.tolist()

with open('F:/SqueezeNet/valid.txt', 'w', encoding='UTF-8') as f:
    for valid_img in valid_data:
        f.write(str(valid_dataset) + '/' + valid_img[0] + ' ' + str(valid_img[1]))
        f.write('\n')
12345678910111213141516171819202122232425262728293031323334353637383940
```

#### 2.2 数据预处理

这里采用到的数据预处理，主要有调整图像大小、随机翻转、归一化等。

```python
import os.path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

transform_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            transforms.Resize(224),  # 调整图像大小为224x224
            transforms.RandomHorizontalFlip(),  # 随机左右翻转图像
            transforms.RandomVerticalFlip(),  # 随机上下翻转图像
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
            transform_BZ  # 执行某些复杂变换操作
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize(224),  # 调整图像大小为224x224
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
            transform_BZ  # 执行某些复杂变换操作
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split(' '), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]

        img_path = os.path.join('', img_path)
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)
12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455565758596061626364
```

#### 2.3 构建模型

```python
import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950515253545556575859606162636465666768697071727374757677787980818283848586878889909192939495
```

#### 2.4 开始训练

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import SqueezeNet
import torchsummary
from dataloader import LoadData
import copy

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = SqueezeNet(num_classes=2).to(device)
# print(model)
#print(torchsummary.summary(model, (3, 224, 224), 1))


# 加载训练集和验证集
train_data = LoadData(r"F:\SqueezeNet\train.txt", True)
train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, pin_memory=True,
                                           shuffle=True, num_workers=0)
test_data = LoadData(r"F:\SqueezeNet\valid.txt", True)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=16, pin_memory=True,
                                           shuffle=True, num_workers=0)


# 编写训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小
    num_batches = len(dataloader)  # 批次数目, (size/batch_size，向上取整)
    print('num_batches:', num_batches)
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率

    for X, y in dataloader:  # 获取图片及其标签
        X, y = X.to(device), y.to(device)
        # 计算预测误差
        pred = model(X)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失

        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新

        # 记录acc与loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= size
    train_loss /= num_batches

    return train_acc, train_loss

# 编写验证函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 测试集的大小
    num_batches = len(dataloader)  # 批次数目, (size/batch_size，向上取整)
    test_loss, test_acc = 0, 0

    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)

            # 计算loss
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)

            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()

    test_acc /= size
    test_loss /= num_batches

    return test_acc, test_loss




# 开始训练

epochs = 20

train_loss = []
train_acc = []
test_loss = []
test_acc = []

best_acc = 0  # 设置一个最佳准确率，作为最佳模型的判别指标


loss_function = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 定义Adam优化器

for epoch in range(epochs):

    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_function, optimizer)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_function)

    # 保存最佳模型到 best_model
    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc
        best_model = copy.deepcopy(model)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    # 获取当前的学习率
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}, Lr:{:.2E}')
    print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss,
                          epoch_test_acc * 100, epoch_test_loss, lr))

# 保存最佳模型到文件中
PATH = './best_model.pth'  # 保存的参数文件名
torch.save(best_model.state_dict(), PATH)

print('Done')
123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100101102103104105106107108109110111112113114115116117118119120121122123
```

![d427d9bb041d471abb947f75032a3b5d](D:/learn/笔记/人工智能/人工智能笔记/image/d427d9bb041d471abb947f75032a3b5d.png)

#### 2.5 结果可视化

```python
import matplotlib.pyplot as plt
#隐藏警告
import warnings
warnings.filterwarnings("ignore")               #忽略警告信息
plt.rcParams['font.sans-serif']    = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
plt.rcParams['figure.dpi']         = 100        #分辨率

epochs_range = range(epochs)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()
123456789101112131415161718192021222324
```

可视化结果如下：
![65442aa0c32ee9fdbc6ffeedada3f161](D:/learn/笔记/人工智能/人工智能笔记/image/65442aa0c32ee9fdbc6ffeedada3f161.png)
可以自行调整学习率以及batch_size，这里我的超参数并没有调整。

### 三、数据集个体预测

```python
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from model import SqueezeNet
import torch

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open("F:\SqueezeNet\data\PALM-Validation400\V0008.jpg")
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
name = ['非病理性近视', '病理性近视']
model_weight_path = r"F:\SqueezeNet\best_model.pth"
model = SqueezeNet(num_classes=2)
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))

    predict = torch.softmax(output, dim=0)
    # 获得最大可能性索引
    predict_cla = torch.argmax(predict).numpy()
    print('索引为', predict_cla)
print('预测结果为：{},置信度为: {}'.format(name[predict_cla], predict[predict_cla].item()))
plt.show()
1234567891011121314151617181920212223242526272829
索引为 1
预测结果为：病理性近视,置信度为: 0.9768268465995789
12
```

![a2049751a39dc7cd492389fd0f280748](D:/learn/笔记/人工智能/人工智能笔记/image/a2049751a39dc7cd492389fd0f280748.png)

## 一、理论基础

## 1.前言

SqueezeNet算法，顾名思义，Squeeze的中文意思是压缩和挤压的意思，所以我们通过算法的名字就可以猜想到，该算法一定是通过解压模型来降低模型参数量的。当然任何算法的改进都是在原先的基础上提升精度或者降低模型参数，因此该算法的主要目的就是在于降低模型参数量的同时保持模型精度。

随着CNN卷积神经网络的研究发展，越来越多的模型被研发出来，而为了提高模型的精度，深层次的模型例如AlexNet和ResNet等得到了大家的广泛认可。但是由于应用场景的需求，许多模型因为参数量太大的原因而无法满足实际的应用场景需求，例如自动驾驶等技术。所以人们开始把注意力放在了轻量级模型上面，

SqueezeNet模型也因此“应时而生”。

SqueezeNet论文中对轻量级模型的优点做了以下几点总结：

● 更高效的分布式训练。服务器间的通信是分布式CNN训练可扩展性的限制因素。对于分布式数据并行训练，通信开销与模型中参数的数量成正比。简而言之，小模型训练更快，因为需要更少的交流。

● 在向客户导出新模型时减少开销。在自动驾驶方面，特斯拉(Tesla)等公司会定期将新车型从服务器复制到客户的汽车上。这种做法通常被称为无线更新。《消费者报告》发现，随着最近的无线更新，特斯拉Autopilot半自动驾驶功能的安全性得到了逐步提高(《消费者报告》，2016年)。然而，今天典型的CNN/DNN模型的无线更新可能需要大量的数据传输。使用AlexNet，这将需要从服务器到汽车的240MB通信。较小的模型需要较少的通信，使频繁更新更可行。

● 可行的FPGA和嵌入式部署。fpga的片内存储器通常小于10MB1，没有片外存储器或存储器。对于推理，足够小的模型可以直接存储在FPGA上，而不会受到内存带宽的瓶颈(Qiu et al .， 2016)，而视频帧则实时通过FPGA。此外，当在专用集成电路(ASIC)上部署cnn时，足够小的模型可以直接存储在芯片上，而更小的模型可以使ASIC适合更小的芯片。

## 2.设计理念

减少模型参数的方法以往就有，一个明智的方法是采用现有的CNN模型并以有损方式压缩它。近年来围绕模型压缩的主题已经出现了一个研究社区，并且已经报道了几种方法。Denton等人的一种相当直接的方法是将奇异值分解(SVD)应用于预训练的CNN模型。Han等人开发了网络剪枝（Network Pruning），从预训练模型开始，将低于某阈值的参数替换为零，形成稀疏矩阵，最后在稀疏CNN上进行几次迭代训练。Han等人将网络修剪与量化(至8位或更少)和霍夫曼编码相结合，扩展了他们的工作，创建了一种称为深度压缩（Deep Compression）的方法，并进一步设计了一种称为EIE的硬件加速器，该加速器直接在压缩模型上运行，实现了显著的加速和节能。

而SqueezeNet算法的主要策略也是压缩策略，在模型压缩上，总共使用了三种方法策略，具体我们下面讲述。

### 2.1 CNN微架构（CNN MicroArchitecture）

同时，SqueezeNet论文作者考虑到随着设计深度CNN卷积神经网络的趋势，手动选择每层滤波器的尺寸变得非常麻烦。所以为了解决这个问题，网上已经有提出了由具有特定固定组织的多个卷积层组成的各种高级构建块或者模块。例如，GoogleNet的论文提出了Inception模块，它由许多不同维度的过滤器组成，通常包括1×11×1和3×33×3，有时候会加上5×55×5，有时加上1×31×3和3×13×1，可能还会加上额外的特设层，形成一个完整的网络。我们上一篇讲解的WideResNet算法中也曾讲解到一个类似的块，即残差块。SqueezeNet论文作者将这种各个模块的特定组织和维度统称为CNN微架构。

![Snipaste_2025-03-14_17-15-15](D:/learn/笔记/人工智能/人工智能笔记/image/Snipaste_2025-03-14_17-15-15.png)

### 2.2 CNN宏架构（CNN MacroArchitecture）

CNN微架构指的是单独的层和模块，而CNN宏架构可以定义为多个模块的系统级组织，形成端到端的CNN体系结构。也许在最近的文献中，最广泛研究的CNN宏观架构主题是网络中深度(即层数)的影响。如VGG12-19层在ImageNet-1k数据集上产生更高的精度。选择跨多层或模块的连接是CNN宏观架构研究的一个新兴领域。例如残差网络（ResNet）和高速公路网络（Highway Network）中都建议采用跳过多层的连接，比如将第3层的激活附加地连接到第6层的激活，我们把这种连接称为旁路连接。ResNet的作者提供了一个34层CNN的A/B比较，有和没有旁路连接（即跳跃连接），实验对比发现添加旁路连接可将ImageNet前5名的精度提高2个百分点。

### 2.3 模型网络设计探索过程

论文作者认为，神经网络(包括深度神经网络DNN和卷积神经网络CNN)有很大的设计空间，有许多微体系结构、宏观体系结构、求解器和其他超参数的选择。很自然，社区想要获得关于这些因素如何影响神经网络准确性的直觉(即设计空间的形状)。神经网络的设计空间探索(DSE)的大部分工作都集中在开发自动化方法来寻找提供更高精度的神经网络架构。这些自动化的DSE方法包括贝叶斯优化(Snoek et al, 2012)、模拟退火(Ludermir et al, 2006)、随机搜索(Bergstra & Bengio, 2012)和遗传算法(Stanley & Miikkulainen, 2002)。值得赞扬的是，这些论文中的每一篇都提供了一个案例，其中提议的DSE方法产生了一个与代表性基线相比具有更高精度的NN架构。然而，这些论文并没有试图提供关于神经网络设计空间形状的直觉。在SqueezeNet论文的后面，作者避开了自动化的方法——相反，作者重构CNN，这样就可以进行有原则的a /B比较，以研究CNN架构决策如何影响模型的大小和准确性。

### 2.4 结构设计策略

SqueezeNet算法的主要目标是构建具有很少参数的CNN架构，同时保证具有其它模型也有的精度。为了实现这一目标，作者总共采用了三种策略来设计CNN架构，具体如下：

**策略1：** 将 3×33×3卷积替换成 1×11×1卷积：通过这一步，一个卷积操作的参数数量减少了99倍；

**策略2：** 减少3×33×3卷积的通道数：一个3×33×3卷积的计算量是3×3×�×�3×3×*M*×*N*（其中M，N分别是输入特征图和输出特征图的通道数），作者认为这样一个计算量过于庞大，因此希望尽可能地将M和N减少以减少参数数量。

**策略3：** 在网络的后期进行下采样，这样卷积层就有了大的激活图。在卷积网络中，每个卷积层产生一个输出激活图，其空间分辨率至少为1x1，通常比1x1大得多。这些激活图的高度和宽度由:(1)输入数据的大小(例如256x256图像)和(2)在CNN体系结构中下采样层的选择。

其中策略1和策略2是关于明智地减少CNN中参数的数量，同时试图保持准确性。策略3是关于在有限的参数预算下最大化准确性。接下来，我们描述Fire模块，这是我们的CNN架构的构建块，使我们能够成功地采用策略1、2和3。

### 2.5 Fire模块

Fire模块包括：挤压卷积层（squeeze convolution），输入扩展层（expand），其中有1×11×1和3×33×3卷积滤波器的混合。在挤压卷积层中只有1×11×1卷积滤波器，而扩展层中混合有1×11×1和3×33×3卷积滤波器。同时该模块中引入了三个调节维度的超参数：

● �1�1*s*1*x*1: squeeze 中 1×11×1卷积滤波器个数 ；

● �1�1*e*1*x*1: expand 中 1×11×1 卷积滤波器个数 ；

● �3�3*e*3*x*3: expand 中 3×33×3 卷积滤波器个数 ；

![Snipaste_2025-03-14_17-16-24](D:/learn/笔记/人工智能/人工智能笔记/image/Snipaste_2025-03-14_17-16-24.png)

![ba9993ccb8894b3dbf90b4fec460eab8affb7b6f986848ef93a7ea43015b30c4](D:/learn/笔记/人工智能/人工智能笔记/image/ba9993ccb8894b3dbf90b4fec460eab8affb7b6f986848ef93a7ea43015b30c4.png)

![Snipaste_2025-03-14_17-17-50](D:/learn/笔记/人工智能/人工智能笔记/image/Snipaste_2025-03-14_17-17-50.png)

## 3.网络结构

![3dfecc9244ef411c80f34fa1701325414deb1a8e20934985849aedb72270b579](D:/learn/笔记/人工智能/人工智能笔记/image/3dfecc9244ef411c80f34fa1701325414deb1a8e20934985849aedb72270b579.png)

● 左图：SqueezeNet ；

● 中图：带简单旁路的 SqueezeNet ；

● 右图：带复杂旁路的 SqueezeNet ；

由图1-2中的左图我们可以看出，SqueezeNet是先从一个独立的卷积层（Conv1）开始，然后经过8个Fire模块（fire2-9），最后以一个卷积层（Conv10）结束。从网络的开始到结束，我们逐渐增加每个fire模块的过滤器数量。在conv1、fire4、fire8和conv10层之后，SqueezeNet以2步长执行最大池化;这些相对较晚的池放置是根据2.4中的策略3。

而简单旁路架构在模块3、5、7和9周围增加了旁路连接，要求这些模块在输入和输出之间学习残差函数。与ResNet一样，为了实现绕过Fire3的连接，将Fire4的输入设置为(Fire2的输出+ Fire3的输出)，其中+运算符是元素加法。这改变了应用于这些Fire模块参数的正则化，并且，根据ResNet，可以提高最终的准确性或训练完整模型的能力。

虽然简单的旁路“只是一根线”，但复杂的旁路包含1x1卷积层的旁路，其滤波器数量设置等于所需的输出通道数量。

**注意：在简单的情况下，输入通道的数量和输出通道的数量必须相同。因此，只有一半的Fire模块可以进行简单的旁路连接，如图2中图所示。当不能满足“相同数量的通道”要求时，我们使用复杂的旁路连接，如图1-2右侧所示。复杂的旁路连接会向模型中添加额外的参数，而简单的旁路连接则不会。**

**完整的SqueezeNet架构如下图所示：**

![19bc13f13fb0464c92428842040eb495b29722d6eb864e2dac90060743c8ab4f](D:/learn/笔记/人工智能/人工智能笔记/image/19bc13f13fb0464c92428842040eb495b29722d6eb864e2dac90060743c8ab4f.png)

## 4.评估分析

比较SqueezeNet和不同模型压缩方法结果图如下：

![4ef622b5e33648a6931fa394321008e72dc00c44e2a245daa593776e75a6f634](D:/learn/笔记/人工智能/人工智能笔记/image/4ef622b5e33648a6931fa394321008e72dc00c44e2a245daa593776e75a6f634.png)

微架构下挤压比率（SR）对模型尺寸和精度的影响以及扩展层中3×33×3过滤器的比例对模型尺寸和精度的影响实验对比图如下所示：

![4265a8f5950245f6a51874adfd06aa0edd604871529145108793b55e03d681a6](D:/learn/笔记/人工智能/人工智能笔记/image/4265a8f5950245f6a51874adfd06aa0edd604871529145108793b55e03d681a6.png)

宏架构中三种结构（普通架构、简单旁支架构和复杂旁支架构）精度对比图如下：

![4dd5e2427886449cb7be15955ccba677bc1ca1985d9640aaabc9194dff9c540d](D:/learn/笔记/人工智能/人工智能笔记/image/4dd5e2427886449cb7be15955ccba677bc1ca1985d9640aaabc9194dff9c540d.png)

## 二、实战

------

## 1.数据预处理

- 导入相关库

In [ ]

```
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math
import random
import os
from paddle.io import Dataset  # 导入Datasrt库
import paddle.vision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
```

In [ ]

```
!pip install openpyxl -i https://pypi.org/simple
```

In [ ]

```
# 解压数据集
!unzip /home/aistudio/data/data23828/training.zip -d /home/aistudio/work/dataset
!unzip /home/aistudio/data/data23828/valid_gt.zip -d /home/aistudio/work/dataset
!unzip /home/aistudio/data/data23828/validation.zip -d /home/aistudio/work/dataset
!unzip /home/aistudio/work/dataset/PALM-Training400/PALM-Training400-Annotation-D&F.zip -d /home/aistudio/work/dataset/PALM-Training400
!unzip /home/aistudio/work/dataset/PALM-Training400/PALM-Training400-Annotation-Lession.zip -d /home/aistudio/work/dataset/PALM-Training400
!unzip /home/aistudio/work/dataset/PALM-Training400/PALM-Training400.zip -d /home/aistudio/work/dataset/PALM-Training400
```

- 数据类别可视化

In [2]

```
import warnings
import matplotlib.font_manager as font_manager
warnings.filterwarnings("ignore")
# 查看数据集各类数量
datapath = "/home/aistudio/work/dataset/PALM-Training400/PALM-Training400"

font_path  = '/home/aistudio/fonts/simfang.ttf'  # 将路径替换为simfang字体的路径
font_prop = font_manager.FontProperties(fname=font_path)
# 初始化计数器
count_H = 0
count_N = 0
count_P = 0

# 遍历数据集文件夹
for filename in os.listdir(datapath):
    # 检查文件名前缀并增加相应计数器的值
    if filename.startswith('H'):
        count_H += 1
    elif filename.startswith('N'):
        count_N += 1
    elif filename.startswith('P'):
        count_P += 1

# 打印各类数据集的数量
print(f'高度近视眼睛的数据集数量: {count_H}')
print(f'正常眼睛的数据集数量: {count_N}')
print(f'病理性近视眼睛的数据集数量: {count_P}')
# 设置类标签和数量
labels = ['H', 'N', 'P']
counts = [count_H, count_N, count_P]

# 创建柱状图
plt.bar(labels, counts)


# 设置标签和标题的字体为DejaVu Sans
plt.xlabel('类别', fontproperties=font_prop)
plt.ylabel('数量', fontproperties=font_prop)
plt.title('数据类别数量', fontproperties=font_prop)

# 显示图形
plt.show()
高度近视眼睛的数据集数量: 26
正常眼睛的数据集数量: 161
病理性近视眼睛的数据集数量: 213
```

![下载 (D:/learn/笔记/人工智能/人工智能笔记/image/下载 (81).png)](image\下载 (81).png)

```
<Figure size 640x480 with 1 Axes>
```

- 数据集类别划分

In [6]

```
# 将训练集划分标签
train_dataset = "/home/aistudio/work/dataset/PALM-Training400/PALM-Training400"
train_list = []
label_list = []


train_filenames = os.listdir(train_dataset)

for name in train_filenames:
    filepath = os.path.join(train_dataset, name)
    train_list.append(filepath)
    if name[0] == 'N' or name[0] == 'H':
        label = 0
        label_list.append(label)
    elif name[0] == 'P':
        label = 1
        label_list.append(label)
    else:
        raise('Error dataset!')


with open('/home/aistudio/work/train.txt','w',encoding='UTF-8') as f:
    i = 0
    for train_img in train_list:
        f.write(str(train_img) + ' ' +str(label_list[i]))
        i += 1
        f.write('\n')
# 将验证集划分标签
valid_dataset = "/home/aistudio/work/dataset/PALM-Validation400"
valid_filenames = os.listdir(valid_dataset)
valid_label = "/home/aistudio/work/dataset/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx"
data = pd.read_excel(valid_label)
valid_data = data[['imgName', 'Label']].values.tolist()

with open('/home/aistudio/work/valid.txt','w',encoding='UTF-8') as f:
    for valid_img in valid_data:
        f.write(str(valid_dataset) + '/' + valid_img[0] + ' ' + str(valid_img[1]))
        f.write('\n')
```

## 2.数据读取

In [3]

```
transform_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            transforms.Resize(224),                  # 调整图像大小为224x224
            transforms.RandomHorizontalFlip(),       #  随机左右翻转图像
            transforms.RandomVerticalFlip(),         # 随机上下翻转图像
            transforms.ToTensor(),                   # 将 PIL 图像转换为张量
            transform_BZ                             # 执行某些复杂变换操作
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize(224),                  # 调整图像大小为224x224
            transforms.ToTensor(),                   # 将 PIL 图像转换为张量
            transform_BZ                             # 执行某些复杂变换操作
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split(' '), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        
        img_path = os.path.join('',img_path)
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)
```

In [5]

```
train_data = LoadData("/home/aistudio/work/train.txt", True)

valid_data = LoadData("/home/aistudio/work/valid.txt", True)
#数据读取
train_loader = paddle.io.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = paddle.io.DataLoader(valid_data, batch_size=64, shuffle=True)
```

## 3.模型构建

In [6]

```
class Fire(nn.Layer):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2D(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2D(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2D(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return paddle.concat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Layer):

    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2D(3, 96, kernel_size=7, stride=2),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2D(3, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2D(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, 1))
        )



    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return paddle.flatten(x, 1)
```

## 4.实例化模型

In [7]

```
import paddle
model = SqueezeNet("1_0", num_classes=4)
params_info = paddle.summary(model,(1, 3, 224, 224))
print(params_info)



W0812 15:21:12.231475   400 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0812 15:21:12.236858   400 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
-------------------------------------------------------------------------------
   Layer (type)         Input Shape          Output Shape         Param #    
===============================================================================
     Conv2D-1        [[1, 3, 224, 224]]   [1, 96, 109, 109]       14,208     
      ReLU-1        [[1, 96, 109, 109]]   [1, 96, 109, 109]          0       
    MaxPool2D-1     [[1, 96, 109, 109]]    [1, 96, 54, 54]           0       
     Conv2D-2        [[1, 96, 54, 54]]     [1, 16, 54, 54]         1,552     
      ReLU-2         [[1, 16, 54, 54]]     [1, 16, 54, 54]           0       
     Conv2D-3        [[1, 16, 54, 54]]     [1, 64, 54, 54]         1,088     
      ReLU-3         [[1, 64, 54, 54]]     [1, 64, 54, 54]           0       
     Conv2D-4        [[1, 16, 54, 54]]     [1, 64, 54, 54]         9,280     
      ReLU-4         [[1, 64, 54, 54]]     [1, 64, 54, 54]           0       
      Fire-1         [[1, 96, 54, 54]]     [1, 128, 54, 54]          0       
     Conv2D-5        [[1, 128, 54, 54]]    [1, 16, 54, 54]         2,064     
      ReLU-5         [[1, 16, 54, 54]]     [1, 16, 54, 54]           0       
     Conv2D-6        [[1, 16, 54, 54]]     [1, 64, 54, 54]         1,088     
      ReLU-6         [[1, 64, 54, 54]]     [1, 64, 54, 54]           0       
     Conv2D-7        [[1, 16, 54, 54]]     [1, 64, 54, 54]         9,280     
      ReLU-7         [[1, 64, 54, 54]]     [1, 64, 54, 54]           0       
      Fire-2         [[1, 128, 54, 54]]    [1, 128, 54, 54]          0       
     Conv2D-8        [[1, 128, 54, 54]]    [1, 32, 54, 54]         4,128     
      ReLU-8         [[1, 32, 54, 54]]     [1, 32, 54, 54]           0       
     Conv2D-9        [[1, 32, 54, 54]]     [1, 128, 54, 54]        4,224     
      ReLU-9         [[1, 128, 54, 54]]    [1, 128, 54, 54]          0       
     Conv2D-10       [[1, 32, 54, 54]]     [1, 128, 54, 54]       36,992     
      ReLU-10        [[1, 128, 54, 54]]    [1, 128, 54, 54]          0       
      Fire-3         [[1, 128, 54, 54]]    [1, 256, 54, 54]          0       
    MaxPool2D-2      [[1, 256, 54, 54]]    [1, 256, 27, 27]          0       
     Conv2D-11       [[1, 256, 27, 27]]    [1, 32, 27, 27]         8,224     
      ReLU-11        [[1, 32, 27, 27]]     [1, 32, 27, 27]           0       
     Conv2D-12       [[1, 32, 27, 27]]     [1, 128, 27, 27]        4,224     
      ReLU-12        [[1, 128, 27, 27]]    [1, 128, 27, 27]          0       
     Conv2D-13       [[1, 32, 27, 27]]     [1, 128, 27, 27]       36,992     
      ReLU-13        [[1, 128, 27, 27]]    [1, 128, 27, 27]          0       
      Fire-4         [[1, 256, 27, 27]]    [1, 256, 27, 27]          0       
     Conv2D-14       [[1, 256, 27, 27]]    [1, 48, 27, 27]        12,336     
      ReLU-14        [[1, 48, 27, 27]]     [1, 48, 27, 27]           0       
     Conv2D-15       [[1, 48, 27, 27]]     [1, 192, 27, 27]        9,408     
      ReLU-15        [[1, 192, 27, 27]]    [1, 192, 27, 27]          0       
     Conv2D-16       [[1, 48, 27, 27]]     [1, 192, 27, 27]       83,136     
      ReLU-16        [[1, 192, 27, 27]]    [1, 192, 27, 27]          0       
      Fire-5         [[1, 256, 27, 27]]    [1, 384, 27, 27]          0       
     Conv2D-17       [[1, 384, 27, 27]]    [1, 48, 27, 27]        18,480     
      ReLU-17        [[1, 48, 27, 27]]     [1, 48, 27, 27]           0       
     Conv2D-18       [[1, 48, 27, 27]]     [1, 192, 27, 27]        9,408     
      ReLU-18        [[1, 192, 27, 27]]    [1, 192, 27, 27]          0       
     Conv2D-19       [[1, 48, 27, 27]]     [1, 192, 27, 27]       83,136     
      ReLU-19        [[1, 192, 27, 27]]    [1, 192, 27, 27]          0       
      Fire-6         [[1, 384, 27, 27]]    [1, 384, 27, 27]          0       
     Conv2D-20       [[1, 384, 27, 27]]    [1, 64, 27, 27]        24,640     
      ReLU-20        [[1, 64, 27, 27]]     [1, 64, 27, 27]           0       
     Conv2D-21       [[1, 64, 27, 27]]     [1, 256, 27, 27]       16,640     
      ReLU-21        [[1, 256, 27, 27]]    [1, 256, 27, 27]          0       
     Conv2D-22       [[1, 64, 27, 27]]     [1, 256, 27, 27]       147,712    
      ReLU-22        [[1, 256, 27, 27]]    [1, 256, 27, 27]          0       
      Fire-7         [[1, 384, 27, 27]]    [1, 512, 27, 27]          0       
    MaxPool2D-3      [[1, 512, 27, 27]]    [1, 512, 13, 13]          0       
     Conv2D-23       [[1, 512, 13, 13]]    [1, 64, 13, 13]        32,832     
      ReLU-23        [[1, 64, 13, 13]]     [1, 64, 13, 13]           0       
     Conv2D-24       [[1, 64, 13, 13]]     [1, 256, 13, 13]       16,640     
      ReLU-24        [[1, 256, 13, 13]]    [1, 256, 13, 13]          0       
     Conv2D-25       [[1, 64, 13, 13]]     [1, 256, 13, 13]       147,712    
      ReLU-25        [[1, 256, 13, 13]]    [1, 256, 13, 13]          0       
      Fire-8         [[1, 512, 13, 13]]    [1, 512, 13, 13]          0       
     Dropout-1       [[1, 512, 13, 13]]    [1, 512, 13, 13]          0       
     Conv2D-26       [[1, 512, 13, 13]]     [1, 4, 13, 13]         2,052     
      ReLU-26         [[1, 4, 13, 13]]      [1, 4, 13, 13]           0       
AdaptiveAvgPool2D-1   [[1, 4, 13, 13]]       [1, 4, 1, 1]            0       
===============================================================================
Total params: 737,476
Trainable params: 737,476
Non-trainable params: 0
-------------------------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 89.22
Params size (MB): 2.81
Estimated Total Size (MB): 92.61
-------------------------------------------------------------------------------

{'total_params': 737476, 'trainable_params': 737476}
```

## 5.开始训练

In [8]

```
epoch_num = 40 #训练轮数
learning_rate = 0.0001 #学习率


val_acc_history = []
val_loss_history = []
bestacc=0

def train(model):
    print('start training ... ')
    # turn into training mode
    model.train()

    opt = paddle.optimizer.Adam(learning_rate=learning_rate,
                                parameters=model.parameters())

    for epoch in range(epoch_num):
        acc_train = []
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1],dtype="int64")
            y_data = paddle.unsqueeze(y_data, 1)
            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc = paddle.metric.accuracy(logits, y_data)
            acc_train.append(acc.numpy())
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
                avg_acc = np.mean(acc_train)
                print("[train] accuracy: {}".format(avg_acc))
            loss.backward()
            opt.step()
            opt.clear_grad()
        
        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(test_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1],dtype="int64")
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc = paddle.metric.accuracy(logits, y_data)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[test] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        if bestacc<acc:
            bestacc=acc
            paddle.save(net.state_dict(), "resnet1_net.pdparams")
        model.train()


epoch: 38, batch_id: 0, loss is: [0.25577748]
[train] accuracy: 0.921875
[test] accuracy/loss: 0.9709821343421936/0.09465523809194565
epoch: 39, batch_id: 0, loss is: [0.11098497]
[train] accuracy: 0.953125
[test] accuracy/loss: 0.9575892686843872/0.13277693092823029
```

## 6.结果可视化

In [9]

```
import matplotlib.pyplot as plt
#隐藏警告
import warnings
warnings.filterwarnings("ignore")               #忽略警告信息

epochs_range = range(epoch_num)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, val_acc_history, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Val Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_loss_history, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Val Loss')
plt.show()
```

![下载 (D:/learn/笔记/人工智能/人工智能笔记/image/下载 (82).png)](image\下载 (82).png)

## 总结

SqueezeNet是一种轻量级卷积神经网络，其设计目标是在保持高准确率的同时，尽量减小模型的大小和计算资源消耗。下面是关于SqueezeNet的总结：

1. 轻量级设计：SqueezeNet采用了一种特殊的结构，即“Fire模块”，通过使用较少的参数来提取丰富的特征。这使得SqueezeNet相对于其他深层网络而言具有更小的模型大小。
2. 参数压缩：SqueezeNet通过使用1x1卷积核来减小参数数量，同时使用通道压缩来减小计算量。这样的设计使得SqueezeNet在计算资源受限的环境中表现出色。
3. 高准确率：尽管SqueezeNet是一种轻量级网络，但它在保持模型小型化的同时，仍能提供相对较高的准确率。通过合理的设计，SqueezeNet能够有效地提取和利用图像的特征信息。
4. 适用场景：由于其小型化的特点，SqueezeNet特别适合在资源有限的环境下使用，如移动设备和嵌入式系统。它可以实现图像分类、目标检测和图像分割等计算密集型任务。

总的来说，SqueezeNet是一种在保持高准确率的同时尽量减小模型大小和计算资源消耗的轻量级网络。它的设计和参数压缩策略使其成为在资源受限环境下进行图像处理任务的有力选择。