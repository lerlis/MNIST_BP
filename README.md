# MNIST_BP
使用BP神经网络识别手写数字集MNIST


## 一、代码功能
1. main.py 主要实现了使用BP神经网络训练模型并绘制损失函数和准确率等曲线
![BP_Evaluation](https://github.com/lerlis/MNIST_BP/blob/main/fig/BP_Evaluation.jpg) 
2. MNIST_By_CNN.py 主要实现了使用CNN网络训练模型并绘制相关曲线
3. model_test.py 可以加载已经保存的模型进行测试并实现可视化
![visual](https://github.com/lerlis/MNIST_BP/blob/main/viaual_BP/batch0.jpg) 
4. read_and_plot.py 存放了一些读取数据和绘图的函数
5. test.py 测试各种功能使用的，没有实际意义

## 二、文件结构
1. data文件夹
   * MNIST 使用torchvision自动下载的数据集
   * self  自己手动下载的数据集
2. fig文件夹
   * 存放了绘制的损失函数和准确率等曲线
3. save_model文件夹
   * 存放了保存的相关模型
4. visual_BP文件夹
   * 存放了使用model_test.py测试模型时绘制的可视化图像，绘制的图像包含所有分类错误的手写数据，只包含少量全部正确的图像
  ```
  c = find_width(t)  # c为绘制图像的长和宽
  if c == -1:
      print('Error, can not plot ')
      print(t)
      continue
  elif acc < 1 or (acc == 1 and random.uniform(0.0, 1.0) < 0.005):
      fig = plt.figure()
  ```
5. visual_CNN文件夹
   * 基本同上
   
