import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

data_root = './assets'
training_data = datasets.MNIST(
    root=data_root,
    train=True,
    download=False,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root=data_root,
    train=False,
    download=False,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print("Shape of X [BatchSize, Channel, Height, Weight]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

figure = plt.figure(figsize=(10, 4))
cols, rows = 5, 2
for i in range(1, cols * rows + 1):
    # 返回元素取值范围为[0-high),size大小的tensor。再通过item取值。
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
