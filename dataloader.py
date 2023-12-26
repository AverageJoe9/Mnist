from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader

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
def dataloader(batch_size):
    # 数据

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in test_dataloader:
        print("Shape of X [BatchSize, Channel, Height, Weight]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    return train_dataloader, test_dataloader
