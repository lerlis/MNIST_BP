import os
import gzip
import numpy as np
import matplotlib.pyplot as plt


def change_one_hot_label(X):
    """
    将标签变为one-hot向量
    :param X:
    :return:
    """
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_data(data_folder, normalize=True, flatten=True, one_hot_label=False):
    """
    读入MNIST数据集
    :param data_folder:
    :param normalize: 将图像的像素值正规化为0.0~1.0
    :param flatten: 是否将图像展开为一维数组
    :param one_hot_label: one_hot_label为True的情况下，标签作为one-hot数组返回
                        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    :return: (训练图像, 训练标签), (测试图像, 测试标签)
    """
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0  # 归一化处理
        x_test = x_test.astype(np.float32) / 255.0

    if one_hot_label:
        y_train = change_one_hot_label(y_train)
        y_test = change_one_hot_label(y_test)

    if flatten:
        x_train = x_train.reshape(60000, 784)  # 60000 * 28 * 28
        x_test = x_test.reshape(10000, 784)

    return (x_train, y_train), (x_test, y_test)


def plot_all(train_losses, train_acces, eval_losses, eval_acces, str):
    """
    绘图，训练集、测试集的损失函数和准确度
    :param train_losses:
    :param train_acces:
    :param eval_losses:
    :param eval_acces:
    :return:
    """
    plt.figure(1)
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.title('train loss')
    plt.savefig('./fig/%s_train_loss.jpg' % str)
    plt.show()
    plt.figure(2)
    plt.plot(np.arange(len(train_acces)), train_acces)
    plt.title('train acc')
    plt.savefig('./fig/%s_train acc.jpg' % str)
    plt.show()
    plt.figure(3)
    plt.plot(np.arange(len(eval_losses)), eval_losses)
    plt.title('test loss')
    plt.savefig('./fig/%s_test loss.jpg' % str)
    plt.show()
    plt.figure(4)
    plt.plot(np.arange(len(eval_acces)), eval_acces)
    plt.title('test acc')
    plt.savefig('./fig/%s_test acc.jpg' % str)
    plt.show()
    plt.figure(5)
    plt.plot(np.arange(len(train_losses)), train_losses, label='train_loss')
    plt.plot(np.arange(len(train_acces)), train_acces, label='train_acc')
    plt.plot(np.arange(len(eval_losses)), eval_losses, label='test loss')
    plt.plot(np.arange(len(eval_acces)), eval_acces, label='test acc')
    plt.title('Evaluation')
    plt.legend()
    plt.savefig('./fig/%s_Evaluation.jpg' % str)
    plt.show()