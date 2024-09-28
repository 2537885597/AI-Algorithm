
import math
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置 numpy 忽略某些警告
np.seterr(divide='ignore', invalid='ignore')


# 生成区间 [a, b] 的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为 m*n，并且设置默认为零矩阵
def makeMatrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)  #添加列表元素
    return np.array(a)


#函数 sigmoid()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 函数 sigmoid 的导函数
def derived_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 构造三层 BP 网络
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层， 输出层的节点数
        self.num_in = num_in + 1  # 增加 一个偏置节点
        self.num_hidden = num_hidden + 1  # 增加 一个偏置节点
        self.num_out = num_out

        # 初始化所有节点的激活状态
        self.active_in = np.array([-1.0] * self.num_in)
        self.active_hidden = np.array([-1.0] * self.num_hidden)
        self.active_out = np.array([1.0] * self.num_out)

        # 创建权重矩阵
        self.weight_hidden = makeMatrix(self.num_in, self.num_hidden)
        self.weight_out = makeMatrix(self.num_hidden, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.weight_hidden[i][j] = random_number(0.1, 0.1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.weight_out[i][j] = random_number(0.1, 0.1)

        # 偏置节点权重
        for j in range(self.num_hidden):
            self.weight_hidden[0][j] = 0.1
        for j in range(self.num_out):
            self.weight_out[0][j] = 0.1

    # 信号前向传播
    def forwardPropagation(self, inputs):
        if len(inputs) != self.num_in - 1:
            raise ValueError('与输入层节点不符')

        # 数据输入层
        self.active_in[1:self.num_in] = inputs

        # 数据在隐藏层的处理
        self.sum_hidden = np.dot(self.weight_hidden.T, self.active_in.reshape(-1, 1))
        self.active_hidden = sigmoid(self.sum_hidden)
        self.active_hidden[0] = -1.0  # 偏置设为-1.0

        # 数据在输出层的处理
        self.sum_out = np.dot(self.weight_out.T, self.active_hidden)
        self.active_out = sigmoid(self.sum_out)
        return self.active_out

    # 误差反向传播
    def backwardPropagation(self, targets, lr):
        """
        targets:教师信号
        lr: 学习率
        """
        if self.num_out == 1:
            targets = np.array(targets)
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 均方误差
        error = (1 / 2) * np.dot((targets.reshape(-1, 1) - self.active_out).T,
                                 (targets.reshape(-1, 1) - self.active_out))

        # 输出层误差项
        self.error_out = (targets.reshape(-1, 1) - self.active_out) * (derived_sigmoid(self.sum_out).reshape(-1, 1))

        # 隐层误差项
        self.error_hidden = np.dot(self.weight_out, self.error_out) * derived_sigmoid(self.sum_hidden)

        # 更新权重值
        # 输出层权重
        self.weight_out += lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T

        # 隐层权重
        self.weight_hidden += lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T

        return error

    # 测试 直接前向传播得出结果
    def test(self, patterns):
        for i in patterns:
            inputs = i[0:self.num_in - 1]
            targets = i[self.num_in - 1:]
            result = self.forwardPropagation(inputs)
            # print(f'输入：{inputs:} -> 结果：{result} = 期望：{targets}')
        return result

    # 训练网络 反向传播调参
    def train(self, patterns, iterations=1000, lr=0.1):
        '''
        lr: 学习速率
        '''
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0:self.num_in - 1]
                targets = p[self.num_in - 1:]
                self.forwardPropagation(inputs)
                error += self.backwardPropagation(targets, lr)
            if i % 100 == 0:
                print(f'误差(能量/损失函数值) {error}\n')
                print(f'输出层误差项{self.error_out} 隐层误差项{self.error_hidden}\n')


# 主程序
def demo():
    # 创建一个神经网络：输入层21个节点（需预测的变量），隐藏层21个节点，输出层21个节点（预测值）
    nn = BPNN(21, 21, 21)

    X = list(np.arange(-1, 1.1, 0.1))
    D = [-0.96, -0.577, -0.0729, 0.017, -0.641, -0.66, -0.11, 0.1336, -0.201, -0.434, -0.5, -0.393, -0.1647, 0.0988,
         0.3072, 0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
    A = X + D

    # 将列表转换为NumPy数组
    # 训练模式
    patterns = np.array([A] * 2)

    # 训练神经网络
    nn.train(patterns, iterations=1000, lr=0.1)

    # 测试神经网络
    d = nn.test(patterns)

    plt.plot(X, D)
    plt.plot(X, d)
    plt.show()


if __name__ == '__main__':
    demo()
