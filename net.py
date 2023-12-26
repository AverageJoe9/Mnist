import torch
from torch import nn
from torchinfo import summary

class Model(nn.Module):
    """模拟LeNet搭建的模型
    """

    def __init__(self) -> None:
        super().__init__()
        # self.flatten = nn.Flatten()
        self.conv_relu_stack = nn.Sequential(
            # 卷积层
            nn.Conv2d(1, 32, kernel_size=3),
            # 激活函数
            nn.ReLU(),
            # 池化层
            nn.MaxPool2d(2),
            # 卷积层
            nn.Conv2d(32, 64, kernel_size=3),
            # 激活函数
            nn.ReLU(),
            # 池化层
            nn.MaxPool2d(2)
        )
        self.dense_relu_stack = nn.Sequential(
            # 全连接层
            nn.Flatten(),
            # 线性层
            nn.Linear(5 * 5 * 64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = self.conv_relu_stack(x)
        x = self.dense_relu_stack(x)
        # x = F.log_softmax(x, dim = 1) # 如果用交叉熵，就不用这个
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device=device)
    print(model)
    summary(model)
