import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 定义 SNN 模型结构
# =======================
class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)       # 28x28 -> 24x24
        self.pool1 = nn.MaxPool2d(2, 2)       # 24x24 -> 12x12
        self.conv2 = nn.Conv2d(6, 16, 5)      # 12x12 -> 8x8
        self.pool2 = nn.MaxPool2d(2, 2)       # 8x8 -> 4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, T=8):
        # 初始化膜电位
        mem_conv1 = torch.zeros(x.size(0), 6, 24, 24, device=device)
        mem_conv2 = torch.zeros(x.size(0), 16, 8, 8, device=device)
        mem_fc1 = torch.zeros(x.size(0), 120, device=device)
        mem_fc2 = torch.zeros(x.size(0), 84, device=device)
        out_sum = torch.zeros(x.size(0), 10, device=device)

        for t in range(T):
            # 卷积层1
            cur1 = self.conv1(x)
            mem_conv1 += cur1
            spike_conv1 = (mem_conv1 >= 1.0).float()
            mem_conv1 = mem_conv1 * (1 - spike_conv1)
            pool1_out = self.pool1(spike_conv1)

            # 卷积层2
            cur2 = self.conv2(pool1_out)
            mem_conv2 += cur2
            spike_conv2 = (mem_conv2 >= 1.0).float()
            mem_conv2 = mem_conv2 * (1 - spike_conv2)
            pool2_out = self.pool2(spike_conv2)

            # 全连接层1
            flat = pool2_out.view(x.size(0), -1)
            cur_fc1 = self.fc1(flat)
            mem_fc1 += cur_fc1
            spike_fc1 = (mem_fc1 >= 1.0).float()
            mem_fc1 = mem_fc1 * (1 - spike_fc1)

            # 全连接层2
            cur_fc2 = self.fc2(spike_fc1)
            mem_fc2 += cur_fc2
            spike_fc2 = (mem_fc2 >= 1.0).float()
            mem_fc2 = mem_fc2 * (1 - spike_fc2)

            # 输出层
            out = self.fc3(spike_fc2)
            out_sum += out / T  # 时间步平均
        return out_sum


# =======================
# 训练与测试
# =======================
def train_and_test():
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    # 模型初始化
    model = SCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(3):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/3], Loss: {total_loss/len(train_loader):.4f}")

    # 测试
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_and_test()
