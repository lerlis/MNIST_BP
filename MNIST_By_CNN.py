from read_and_plot_func import *
import torch
import torch.nn as nn
import torchvision
import torch.utils.data.dataloader as dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def data_reader(path, train_size, test_size):
    """
    加载MNIST数据集，分训练集和测试集返回标签、数据
    :param path:
    :return:
    """
    train_set = torchvision.datasets.MNIST(
        root=path, train=True,
        transform=torchvision.transforms.ToTensor(), download=True
    )
    train_loader = dataloader.DataLoader(
        dataset=train_set, shuffle=True, batch_size=train_size
    )
    """
    dataloader返回（images,labels）
    其中，
    images维度：[batch_size,1,28,28]
    labels：[batch_size]，即和图片对应
    shuffle: 打乱数据
    """
    test_set = torchvision.datasets.MNIST(
        root=path, train=False,
        transform=torchvision.transforms.ToTensor(), download=True
    )
    test_loader = dataloader.DataLoader(
        dataset=test_set, shuffle=False, batch_size=test_size
    )
    return train_loader, test_loader


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(  # input shape (1,28,28)
                in_channels=1, out_channels=16, kernel_size=5,
                stride=1, padding=2
            ),  # output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape (16,14,14)
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5,
                stride=1, padding=2
            ),  # output shape (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape (32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, img):
        img = self.CNN1(img)
        img = self.CNN2(img)
        img = img.reshape(img.size(0), -1)
        out = self.out(img)
        return out


def train_model(train_data, test_data, iterations, model, model_criterion, model_optimizer):
    """
    模型训练和评估函数，完成模型训练的整个过程
    :param train_data: 训练用数据集
    :param test_data: 测试用数据集
    :param iterations: 训练迭代的次数
    :param model: 神经网络模型
    :param model_criterion: 损失函数
    :param model_optimizer: 反向传播优化函数
    :return:
    """
    model_train_losses = []
    model_train_acces = []
    model_eval_losses = []
    model_eval_acces = []
    for epoch in range(iterations):
        # 网络训练
        train_loss = 0
        train_acc = 0
        model.train()
        for i, data in enumerate(train_data):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = model_criterion(out, label)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
        model_train_losses.append(train_loss / len(train_data))
        model_train_acces.append(train_acc / len(train_data))

        # 网络评估
        eval_loss = 0
        eval_acc = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_data):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                out = model(img)
                loss = model_criterion(out, label)

                eval_loss += loss.item()

                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / img.shape[0]
                eval_acc += acc
            model_eval_losses.append(eval_loss / len(test_data))
            model_eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(epoch+1, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data)))
    return model_train_losses, model_train_acces, model_eval_losses, model_eval_acces


if __name__ == "__main__":
    # 获取数据
    train_load, test_load = data_reader('./data', train_size=64, test_size=64)  # 使用torchvision库函数读取数据
    train, test = load_data('./data/self')  # 从本地读取数据，暂时没有使用
    # 构建网络、损失函数
    epochs = 25  # 迭代次数
    learning_rate = 0.01  # 学习率
    train_ID = 're'
    model2 = CNNModel()
    model2 = model2.to(device)
    print(model2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)
    # 训练
    train_losses, train_acces, eval_losses, eval_acces = train_model(
        train_data=train_load, test_data=test_load, iterations=epochs,
        model=model2, model_criterion=criterion, model_optimizer=optimizer
    )
    state = {
        'model': model2.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, './save_model/CNN_%s.pth' % train_ID)
    # 绘图
    plot_all(train_losses, train_acces, eval_losses, eval_acces, 'CNN')
