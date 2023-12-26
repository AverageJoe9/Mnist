import torch
import matplotlib.pyplot as plt

from dataloader import test_data

model_name = 'p2.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ' + str(device))

model = torch.load('./models/' + model_name).to(device=device)

figure = plt.figure(figsize=(10, 4))
cols, rows = 5, 2
for i in range(1, cols * rows + 1):
    # 返回元素取值范围为[0-high),size大小的tensor。再通过item取值。
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[idx]
    figure.add_subplot(rows, cols, i)
    pred = model(img.reshape(1, 1, 28, 28).to(device))
    pred = torch.argmax(pred).item()
    label=str(label)
    pred=str(pred)
    plt.title('label:' + label + ' ' + 'pred:' + pred + ' ' + ('right' if label == pred else 'wrong'))
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()