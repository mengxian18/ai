import numpy as np
import pandas as pd 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import DenseNet169_Weights

# 设置训练和测试数据集路径
train_dir = 'D:/code/data/white'
test_dir = 'D:/code/data/test'

# 修改为使用 weights 参数
model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, 3)
best_val_loss = 100

# 数据预处理
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
full_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# 早停策略参数
patience_counter = 0
patience = 5

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    for epoch in range(30):
        model.train()
        running_loss = 0.0

        # 训练步骤
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算验证损失
        def calculate_validation_loss(model, val_loader, criterion, device):
            model.eval()  
            total_loss = 0.0
            total_batches = len(val_loader)

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)  
                    total_loss += loss.item()        

            average_loss = total_loss / total_batches
            return average_loss

        val_loss = calculate_validation_loss(model, val_loader, criterion, device)

        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving model with validation loss: {val_loss}")
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved successfully.")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    try:
        model.load_state_dict(torch.load('best_model.pth'))
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please check if the training was successful.")
    else:
        model.to(device)
        model.eval()

        # 测试数据预处理
        test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 自定义测试数据集类
        class TestDataset(Dataset):
            def __init__(self, image_folder, transform=None):
                self.image_folder = image_folder
                self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
                self.transform = transform

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                img_path = os.path.join(self.image_folder, self.image_files[idx])
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, self.image_files[idx] 

        # 创建测试数据集和数据加载器
        test_dataset = TestDataset(test_dir, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # 类别名称映射
        class_names = ['plastic', 'glass', 'metal']

        # 进行测试并保存结果
        predictions = []
        output_folder = 'annotated_images'
        os.makedirs(output_folder, exist_ok=True)
        font = ImageFont.load_default()
        with torch.no_grad():
            for inputs, paths in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                for img_path, label in zip(paths, predicted.cpu().numpy()):
                    img_id = os.path.basename(img_path).split('.')[0]
                    predictions.append({'Image_ID': img_id, 'Label': label})

                    # 打开原始图像
                    original_image = Image.open(os.path.join(test_dir, img_path))
                    draw = ImageDraw.Draw(original_image)
                    class_name = class_names[label]
                    # 在图像上标注类别
                    draw.text((10, 10), class_name, fill=(255, 0, 0), font=font)
                    # 保存标注后的图像
                    output_path = os.path.join(output_folder, img_id + '_annotated.jpg')
                    original_image.save(output_path)

        submission = pd.DataFrame(predictions)
        submission.to_csv('21f1002275.csv', index=False)    