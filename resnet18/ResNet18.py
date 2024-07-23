import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
# from PIL import Image
import torch.nn.functional as F



transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5,0.5,0.5],
                                    std=[0.5,0.5,0.5]
                                ),
                                transforms.Resize((224, 224))
                               ])

training_data = datasets.CIFAR10(
    root="dataset",
    train=True,
    download=True,
    transform=transform,
)

testing_data = datasets.CIFAR10(
    root="dataset",
    train=False,
    download=True,
    transform=transform,
)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
#         print('shape of x: {}'.format(x.shape))
        out = self.layer(x)
#         print('shape of out: {}'.format(out.shape))
#         print('After shortcut shape of x: {}'.format(self.shortcut(x).shape))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

#         out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out
    
# 保持数据集和测试机能完整划分
batch_size=100
train_data = DataLoader(dataset=training_data,batch_size=batch_size,shuffle=True,drop_last=True)
test_data = DataLoader(dataset=testing_data,batch_size=batch_size,shuffle=True,drop_last=True)

images,labels = next(iter(train_data))
print(images.shape)
img = utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(BasicBlock, num_classes=10).to(device)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(len(train_data))
print(len(test_data))
epochs = 100
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    model.train()
    print("Epoch {}/{}".format(epoch+1,epochs))
    print("-"*10)
    for X_train,y_train in train_data:
        # X_train,y_train = torch.autograd.Variable(X_train),torch.autograd.Variable(y_train)
        X_train,y_train = X_train.to(device), y_train.to(device)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss = cost(outputs,y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0
    test_loss = 0
    model.eval()
    for X_test,y_test in test_data:
        # X_test,y_test = torch.autograd.Variable(X_test),torch.autograd.Variable(y_test)
        X_test,y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        loss = cost(outputs,y_test)
        _,pred = torch.max(outputs.data,1)
        testing_correct += torch.sum(pred == y_test.data)
        test_loss += loss.item()
    print("Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is::{:.4f} Test Accuracy is:{:.4f}%".format(
        running_loss/len(training_data), 100*running_correct/len(training_data),
        test_loss/len(testing_data),
        100*testing_correct/len(testing_data)
    ))

