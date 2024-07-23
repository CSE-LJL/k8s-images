#!/usr/bin/env python3

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
import time
epochs = 100
batch_size = 64

def train_single_gpu():
    torch.manual_seed(1138)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50(num_classes=10).to(device)

    cifar_path="../data"
    transform = transforms.Compose(
        [
            transforms.Resize(256), #将输入图像的较短边调整到 256 像素
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=cifar_path, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=cifar_path, train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-8)
    
    net.train()
    
    gpu_start_time = time.time()
    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到 GPU 上
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
                running_loss = 0.0


if __name__ == "__main__":
    train_single_gpu()

