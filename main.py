from read_and_plot_func import *
import torch
import torch.nn as nn
import torchvision
import torch.utils.data.dataloader as dataloader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def data_tf(image):
    """
    数据归一化并将维度展开：[28,28]->[1，784]
    :param image:
    :return:
    """
    img = np.array(image, dtype='float32') / 255
    # img = (img - 0.5) / 0.5
    img = img.reshape((-1,))
    img = torch.from_numpy(img)
    return img


def data_reader(path, train_size, test_size):
    """
    加载MNIST数据集，分训练集和测试集返回标签、数据
    :param path:
    :return:
    """
    train_set = torchvision.datasets.MNIST(
        root=path, train=True,
        transform=data_tf, download=True
    )
    train_loader = dataloader.DataLoader(
        dataset=train_set, shuffle=True, batch_size=train_size
    )
    """
    dataloader返回（images,labels）
    其中，
    images维度：[batch_size,1,28,28]->[batch_size, 28*28]
    labels：[batch_size]，即和图片对应
    shuffle: 打乱数据
    """
    # print(train_set)
    test_set = torchvision.datasets.MNIST(
        root=path, train=False,
        transform=data_tf, download=True
    )
    test_loader = dataloader.DataLoader(
        dataset=test_set, shuffle=False, batch_size=test_size
    )
    # print(test_set)

    # firstImg, firstImg_label = train_set[0]
    # print(firstImg.shape)
    # print(firstImg_label)

    # batch_images, batch_label = next(iter(train_loader))
    # print(batch_images.shape)
    # print(batch_label.shape)
    return train_loader, test_loader


class BPNNMdel(torch.nn.Module):
    def __init__(self):
        super(BPNNMdel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(400, 200), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(200, 100), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(100, 10))  # 输出维度必须大于标签的维度，即最好大于分类数，否则报错

    def forward(self, img):
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img


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
    # for i, data in enumerate(train_load):
    #     images, labels = data
    #     print(images)
    #     print(labels)
    #     print(i)
    # x_train, y_train = train
    # print(y_train[5])
    # 构建网络、损失函数
    epochs = 25  # 迭代次数
    learning_rate = 0.01  # 学习率
    Model = BPNNMdel()
    Model = Model.to(device)
    print(Model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Model.parameters(), lr=learning_rate)
    # 训练
    train_ID = 're'
    train_losses, train_acces, eval_losses, eval_acces = train_model(
        train_data=train_load, test_data=test_load, iterations=epochs,
        model=Model, model_criterion=criterion, model_optimizer=optimizer
    )
    state = {
        'model': Model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, './save_model/BP_%s.pth' % train_ID)
    # 绘图
    plot_all(train_losses, train_acces, eval_losses, eval_acces, 'BP')
