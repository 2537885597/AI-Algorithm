# import math
# import string

# import numpy.matlib
# import numpy as np

# np.seterr(divide='ignore', invalid='ignore')

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import font_manager

# import pandas as pd
# import random

# #生成区间[a, b]的随机数
# def random_number(a, b):
#     return (b-a)*random.random() + a

# #生成一个矩阵，大小为m*n，并且设置默认为零矩阵
# def makeMatrix(m, n, fill=0.0):
#     a = []
#     for i in range(m):
#         a.append([fill]*n)
#     return np.array(a)

# #函数sigmoid()
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# #函数sigmoid的导函数
# def derived_sigmoid(x):
#     return sigmoid(x) * (1 - sigmoid(x))

# #构造三层BP网络
# class BPNN:
#     def __init__(self, num_in, num_hidden, num_out):
#         #输入层，隐藏层， 输出层的节点数
#         self.num_in = num_in + 1  #增加一个偏置节点
#         self.num_hidden = num_hidden + 1 #增加一个偏置节点
#         self.num_out = num_out

#         #激活神经网络的所有节点（向量）？
#         self.active_in = np.array([-1.0]*self.num_in)
#         self.active_hidden = np.array([-1.0]*self.num_hidden)
#         self.active_out = np.array([1.0]*self.num_out)

#         #创建权重矩阵
#         self.weight_in = makeMatrix(self.num_in, self.num_hidden)
#         self.weight_out = makeMatrix(self.num_hidden, self.num_out)

#         #对权值矩阵赋初值
#         for i in range(self.num_in):
#             for j in range(self.num_hidden):
#                 self.weight_in[i][j] = random_number(0.1, 0.1)
#         for i in range(self.num_hidden):
#             for j in range(self.num_out):
#                 self.weight_out[i][j] = random_number(0.1, 0.1)

#         #偏差
#         for j in range(self.num_hidden):
#             self.weight_in[0][j] = 0.1
#         for j in range (self.num_out):
#             self.weight_out[0][j] = 0.1

#         #最后建立动量因子？
#         self.ci = makeMatrix(self.num_in, self.num_hidden)
#         self.co = makeMatrix(self.num_hidden, num_out)

#     #信号前向传播
#     def forwardPropagation(self, inputs):
#         if len(inputs) != self.num_in -1:
#             raise ValueError('与输入层节点不符')
        
#         #数据输入层
#         self.active_in[1:self.num_in] = inputs

#         #数据在隐藏层的处理
#         self.sum_hidden = np.dot(self.weight_in.T, self.active_in.reshape(-1, 1)) #点乘？
#         self.active_hidden = sigmoid(self.sum_hidden)
#         self.active_hidden[0] = -1.0

#         #数据在输出层的处理
#         self.sum_out = np.dot(self.weight_out.T, self.active_hidden) #点乘
#         self.active_out = sigmoid(self.sum_out)
#         return self.active_out

#     #误差反向传播
#     def backwardPropagation(self, targets, lr, m):
#         """
#         lr:学习率
#         """
#         if self.num_out == 1:
#             targets = [targets]
#         if len(targets) != self.num_out:
#             raise ValueError('与输出层节点数不符')
#         targets = np.array(targets)
#         #误差  --reshape(row, column) -1代表不指定 --T代表转置矩阵
#         error = (1/2) * np.dot((targets.reshape(-1, 1)) - self.active_out.T, (targets.reshape(-1,1)) - self.active_out)

#         #输出误差信号？
#         self.error_out = (targets.reshape(-1, 1) - self.active_out) * derived_sigmoid(self.sum_out)

#         #隐层误差信号?
#         self.error_hidden = np.dot(self.weight_out, self.error_out) * derived_sigmoid(self.sum_hidden)

#         #更新权重值?
#         #隐层
#         self.weight_out = self.weight_out + lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T + m * self.co
#         self.co = lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T

#         #输入层
#         self.weight_in = self.weight_in + lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T + m * self.ci
#         self.ci = lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T

#         return error
    
#     #测试
#     def test(self, patterns):
#         for i in patterns:
#             print(i[0:self.num_in - 1], '->', self.forwardPropagation(i[0:self.num_in - 1]))

#         return self.forwardPropagation(i[0:self.num_in - 1])
    
#     #权值显示
#     def weights(self):
#         print("输入层权重")
#         print(self.weight_in)
#         print("输出层权重")
#         print(self.weight_out)
    
#     #训练
#     def train(self, pattern, itera=100, lr=0.2, m=0.1):
#         for i in range(itera):
#             error = 0.0
#             for j in pattern:
#                 inputs = j[0:self.num_in - 1]
#                 targets = j[self.num_in - 1]
#                 self.forwardPropagation(inputs)
#                 error = error + self.backwardPropagation(targets, lr, m)
#             if i % 10 == 0:
#                 print("#########################误差 %-.5f####################第%d次迭代"%(error, i))

# #实例
# X = list(np.arange(-1, 1.1, 0.1))
# D = [-0.96, -0.577, -0.0729, 0.017, -0.641, -0.66, -0.11, 0.1336, -0.201, -0.434, -0.5, -0.393, -0.1647, 0.0988, 0.3072, 0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
# A = X + D
# patt = np.array([A] * 2)
# #创建神经网络
# n = BPNN(21, 21, 1)
# #训练神经网络
# n.train(patt)
# #测试神经网络
# d = n.test(patt)
# #查阅权重
# n.weights()

# plt.plot(X, D)
# plt.plot(X, d)
# plt.show()

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
    return np.full((m, n), fill)

# 函数 sigmoid()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 函数 sigmoid 的派生函数
def derived_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 构造三层 BP 网络
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层， 输出层的节点数
        self.num_in = num_in + 1  # 增加 一个偏置节点
        self.num_hidden = num_hidden + 1  # 增加 一个偏置节点
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）？
        self.active_in = np.array([-1.0] * self.num_in)
        self.active_hidden = np.array([-1.0] * self.num_hidden)
        self.active_out = np.array([1.0] * self.num_out)

        # 创建权重矩阵
        self.weight_in = makeMatrix(self.num_in, self.num_hidden)
        self.weight_out = makeMatrix(self.num_hidden, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.weight_in[i][j] = random_number(-0.1, 0.1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.weight_out[i][j] = random_number(-0.1, 0.1)

        # 偏置
        for j in range(self.num_hidden):
            self.weight_in[0][j] = random_number(-0.1, 0.1)
        for j in range(self.num_out):
            self.weight_out[0][j] = random_number(-0.1, 0.1)

        # 最后建立动量因子？
        self.ci = makeMatrix(self.num_in, self.num_hidden)
        self.co = makeMatrix(self.num_hidden, self.num_out)

    # 信号前向传播
    def forwardPropagation(self, inputs):
        if len(inputs) != self.num_in - 1:
            raise ValueError('与输入层节点不符')

        # 数据输入层
        self.active_in[1:self.num_in] = inputs

        # 数据在隐藏层的处理
        self.sum_hidden = np.dot(self.weight_in.T, self.active_in)
        self.active_hidden = sigmoid(self.sum_hidden)
        self.active_hidden[0] = -1.0

        # 数据在输出层的处理
        self.sum_out = np.dot(self.weight_out.T, self.active_hidden)
        self.active_out = sigmoid(self.sum_out)
        return self.active_out

    # 误差反向传播
    def backwardPropagation(self, targets, lr, m):
        """
        lr: 学习率
        m: 动量因子
        """
        targets = np.array([targets])
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符')

        # 误差
        error = 0.5 * np.sum((targets - self.active_out) ** 2)

        # 输出误差信号？
        self.error_out = (targets - self.active_out) * derived_sigmoid(self.sum_out)

        # 隐层误差信号?
        self.error_hidden = np.dot(self.weight_out, self.error_out) * derived_sigmoid(self.sum_hidden)

        # 更新权重值?
        # 隐层
        self.weight_out += lr * np.outer(self.error_out, self.active_hidden).T + m * self.co
        self.co = lr * np.outer(self.error_out, self.active_hidden).T

        # 输入层
        self.weight_in += lr * np.outer(self.error_hidden, self.active_in).T + m * self.ci
        self.ci = lr * np.outer(self.error_hidden, self.active_in).T

        return error

    # 测试
    def test(self, patterns):
        results = []
        for i in patterns:
            inputs = i[0:self.num_in - 1]
            targets = i[self.num_in - 1:]
            result = self.forwardPropagation(inputs)
            results.append(result)
            print(inputs)
            print('->', result)
            print('=', targets)
            print('---')
        return results
    
    # 训练网络
    def train(self, patterns, iterations=1000, lr=0.1, m=0.1):
        # lr: 学习速率
        # m: 动量因子
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0:self.num_in - 1]
                targets = p[self.num_in - 1:]
                self.forwardPropagation(inputs)
                error += self.backwardPropagation(targets, lr, m)
            if i % 100 == 0:
                print('误差 %-.5f' % error)

# 主程序
def demo():
    # 创建一个神经网络：输入层3个节点，隐藏层5个节点，输出层2个节点
    nn = BPNN(3, 5, 2)

    # 训练模式
    patterns = [
        np.array([0, 0, 0, 0, 0]),
        np.array([0, 0, 1, 0, 1]),
        np.array([0, 1, 0, 1, 0]),
        np.array([0, 1, 1, 1, 1]),
        np.array([1, 0, 0, 1, 0]),
        np.array([1, 0, 1, 1, 1]),
        np.array([1, 1, 0, 1, 1]),
        np.array([1, 1, 1, 1, 0])
    ]

    # 训练神经网络
    nn.train(patterns, iterations=1000, lr=0.1, m=0.1)

    # 测试神经网络
    nn.test(patterns)

if __name__ == '__main__':
    demo()