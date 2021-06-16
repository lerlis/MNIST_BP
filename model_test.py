import torch
from main import BPNNMdel
from MNIST_By_CNN import CNNModel
import MNIST_By_CNN
import main
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def find_width(num):
    """
    定义绘图时每行每列的子图个数
    :param num:
    :return:
    """
    for i in range(1, num+1):
        if i * i == num:
            return [i, i]
        elif i * (i + 1) < num < (i + 1) ** 2:
            return [i+1, i+1]
        elif i * i < num < i * (i + 1):
            return [i+1, i]
    return -1


def test(model, test_data):
    """
    评估模型的准确度，并进行可视化
    :param model:
    :param test_data:
    :return:
    """
    eval_acc = 0
    model.eval()
    num_corrects = 0
    sum_t = 0
    with torch.no_grad():
        for i, data in enumerate(test_data):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            out = model(img)

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            num_corrects += num_correct
            acc = float(num_correct) / img.shape[0]
            eval_acc += acc
            t = len(label)
            sum_t += t
            print('Batch is %s :' % i, 'Accuracy: {}/{} ({:.3f}%)'.format(
                num_correct, t, 100. * acc
            ))

            show_pic = img.reshape(t, 28, 28).cpu()
            c = find_width(t)
            if c == -1:
                print('Error, can not plot ')
                print(t)
                continue
            elif acc < 1 or (acc == 1 and random.uniform(0.0, 1.0) < 0.005):
                fig = plt.figure()
                for j in range(t):
                    plt.subplot(c[0], c[1], j + 1)
                    plt.tight_layout()
                    plt.imshow(show_pic[j], cmap='gray', interpolation='none')
                    plt.title("Prediction: {} \n Label: {}".format(
                        out.data.max(1, keepdim=True)[1][j].item(), label[j]
                    ))
                    plt.xticks([])
                    plt.yticks([])
                plt.savefig('./visual_BP/batch%s.jpg' % i)
                plt.show()
        print(
            'Accuracy: {}/{} ({:.3f}%)'.format(
                num_corrects, sum_t, 100. * eval_acc / len(test_data)
            )
        )
    return pred, float(eval_acc) / len(test_data)


if __name__ == '__main__':
    learning_rate = 0.01
    path = './save_model/BP_2.pth'
    checkpoint = torch.load(path)

    model = BPNNMdel()
    # model = CNNModel()
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])

    data_path = './data'
    _, test_data = main.data_reader(data_path, train_size=64, test_size=9)
    # _, test_data = MNIST_By_CNN.data_reader(data_path, train_size=64, test_size=9)
    pred, accuracy = test(model, test_data)
