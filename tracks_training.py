# .
# ├── tracks_training.py
# ├── parameters
# │   ├── resnet18_e1.pth
# │   ├── resnet18_e2.pth
# │   └── resnet18_e3.pth
# ├── training_tracks
# │   ├── 0
# │   │   ├── 1.jpg
# │   │   ├── 2.jpg
# │   │   └── 3.jpg
# │   ├── 1
# │   │   ├── 1.jpg
# │   │   ├── 2.jpg
# │   │   └── 3.jpg
# │   └── 2
# │       ├── 1.jpg
# │       ├── 2.jpg
# │       └── 3.jpg
# └── testing_tracks
#     ├── 0
#     │   ├── 1.jpg
#     │   ├── 2.jpg
#     │   └── 3.jpg
#     ├── 1
#     │   ├── 1.jpg
#     │   ├── 2.jpg
#     │   └── 3.jpg
#     └── 2
#         ├── 1.jpg
#         ├── 2.jpg
#         └── 3.jpg

import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(in_features=512, out_features=3)
        self.norm = nn.BatchNorm1d(num_features=512)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        f512 = self.resnet(x)
        f512 = f512.view(-1, 512)
        f512 = self.norm(f512)
        f512 = self.relu(f512)
        f3 = self.fc(f512)
        return f3


def train_model(batch_size: int, start_epoch: int, end_epoch: int, learning_rate: float, model_name: str):
    train_set = torchvision.datasets.ImageFolder(root='./training_tracks', transform=Transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    net = Net().to(Device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if start_epoch >= 1:
        net.load_state_dict(torch.load(f"parameters/{model_name}_e{start_epoch}.pth"))

    for epoch in range(start_epoch + 1, end_epoch + 1):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(Device), data[1].to(Device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"epoch {epoch}, batch {i}, loss = {loss.item()}")
        torch.save(net.state_dict(), f"parameters/{model_name}_e{epoch}.pth")


def t_model(batch_size: int, epoch: int, model_name: str):
    test_set = torchvision.datasets.ImageFolder(root='./testing_tracks', transform=Transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    net = Net().to(Device)
    if epoch >= 1:
        net.load_state_dict(torch.load(f"parameters/{model_name}_e{epoch}.pth"))

    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(Device), data[1].to(Device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of epoch {epoch} = {100 * correct // total} %")


if __name__ == '__main__':
    print(Device)

    if not os.path.exists("./parameters"):
        os.mkdir("parameters")

    for e in range(19, 101):
        start_time = time.time()
        train_model(batch_size=128, start_epoch=e - 1, end_epoch=e, learning_rate=0.01, model_name="resnet18")
        end_time = time.time()
        print(f"Training time for epoch {e} = {end_time - start_time:.2f}s")
        t_model(batch_size=128, epoch=e, model_name="resnet18")

# 初始化部分，放在 while 外面

from tracks_training import Net

device = torch.device("cpu")
net = Net().to(device)
net.train(False)
net.load_state_dict(torch.load(f"parameters/resnet18_e10.pth", map_location=device))


# 调用模型，输入原始图片，输出运动状态

def auto_drive_with_nn(image: numpy.ndarray):
    with torch.no_grad():
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)  # 缩放成 224 * 224
        image = np.transpose(image, (2, 0, 1))  # 将RGB通道移到最前面
        image = image.reshape(1, 3, 224, 224)  # 重塑成 batch_size * channel * width * height 形式
        image = torch.tensor(image / 255, dtype=torch.float).to(device)  # 将值从 0-255 映射到 0-1 并转成 tensor
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        moving_status = predicted.item()
        return moving_status


# while 循环中调用方式
auto_drive_with_nn(input_frame)