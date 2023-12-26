import torch
from torch import nn
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torch.nn.functional as F

from net import Model
from dataloader import dataloader

# 参数
model_name = 'p2.pth'
epochs = 10
batch_size = 64
learning_rate = 1e-3

# 判断是否有cuda环境，如果有则使用cuda环境
#  当torch._C._cuda_getDeviceCount() > 0，重启有时会解决问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ' + str(device))

train_dataloader, test_dataloader = dataloader(batch_size)
model = Model().to(device=device)
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数，该损失函数不需要softmax层
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    batchs = len(dataloader)
    train_acc, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算损失值
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()  # 梯度置0
        loss.backward()  # 求梯度
        optimizer.step()  # 根据梯度计算w的值

        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f} [{current:5d}/{size:>5d}]")
    train_acc /= size
    train_loss /= batchs
    return train_acc, train_loss


def validate(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    batchs = len(dataloader)
    validate_acc, validate_loss = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validate_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            validate_loss += loss_fn(pred, y).item()
    validate_acc /= size
    validate_loss /= batchs
    return validate_acc, validate_loss


res = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for i in range(epochs):
    epoch_train_acc, epoch_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    epoch_validate_acc, epoch_validate_loss = validate(test_dataloader, model)
    print(f'Epoch:{i:3d}, Train_acc:{epoch_train_acc:.4f}, Train_loss:{epoch_train_loss:.4f}, ')
    res["train_acc"].append(epoch_train_acc)
    res["train_loss"].append(epoch_train_loss)
    res["val_acc"].append(epoch_validate_acc)
    res["val_loss"].append(epoch_validate_loss)

# 设置字体为雅黑，unicode显示负号，像素dpi为100
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
epoch_range = range(epochs)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.plot(epoch_range, res['train_acc'], label='Training Accuracy')
plt.plot(epoch_range, res['val_acc'], label='Validation Accuracy')
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epoch_range, res['train_loss'], label='Training Loss')
plt.plot(epoch_range, res['val_loss'], label='Validation Loss')
plt.legend(loc="upper right")
plt.title('Training and Validation Loss')
plt.show()

print("最终准确率", res["val_acc"][-1])
torch.save(model, './models/' + model_name)
