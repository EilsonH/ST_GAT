#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor  # 这里必须加.因为这些文件都是在下级目录中，不在


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# processor就是传入变量，这里是'-c', 'config/st_gcn/kinetics-skeleton/test.yaml'
class REC_Processor(Processor):  # 继承自processor类，没有定义构造函数，所以构造函数就是直接继承processor中的
    """
        Processor for Skeleton-based Action Recgnition
    """

    # 原本recognition里面没有重写构造函数，而是直接继承了该方法
    # 不过直接抄过来也没有问题
    # 在构造函数中，并没有调用命令行
    def __init__(self, argv=None):
        self.load_arg(argv)  # 继承自io
        # self.arg.freeze_graph_until=10
        self.init_environment()  # 在io中调用，在processor中继承,如果设置断点，必须在processor中设置
        self.load_model()  # 载入st-gcn模型，只载入一次
        self.load_weights()  # 在io中定义，一直继承到本文件
        self.gpu()  # 在io中定义
        self.load_data()  # 在processor中定义
        self.load_optimizer()  # 在本文件中定义

    # self.start()
    def load_model(self):
        # self.io在io中定义的，是一个容器对象
        # 这里self.io.load_model的参数就是模型的位置(字符串)和四个属性值(字典)，都是从yaml文件中获取的
        # 得到的self.model是个net.st_gcn.model对象，通过调用其__class__内建方法可以查看
        self.model = self.io.load_model(self.arg.model,  # self.arg.model就是st-gcn,这里是字符串形式
                                        **(
                                        self.arg.model_args))  # arg.model_args就是那4个参数，是字典形式，变量前面加*表示元组形式的参数，**表示字典形式的参数
        self.model.apply(weights_init)  # 执行一次权重初始化
        self.loss = nn.CrossEntropyLoss()  # 设置损失函数为交叉熵

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):  # 学习率下降策略
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    # def update_graph_freeze(self, epoch):
    #     graph_requires_grad = (epoch > self.arg.freeze_graph_until)
    #     self.print_log('Graphs are {} at epoch {}'.format('learnable' if graph_requires_grad else 'frozen', epoch + 1))
    #     for param in self.param_groups['graph']:
    #         param.requires_grad = graph_requires_grad

    # 在processor.py中通过start调用
    def train(self, epoch):  # 完全覆盖父类方法，每个epoch都会执行一次
        # self.model是个net.st_gcn.model对象
        # 现在self.model是经过nn.Parallel包装过的模型，相当于套了一层外壳，通过model.module来获取真实模型
        if self.arg.freeze_graph_until >= 0:  # 只要把这个值设置为小于0，就不会进行学习
            print(self.model.module._parameters['A'])
            if epoch == self.arg.freeze_graph_until:
                self.model.module.graph_learn()
                print(self.model.module._parameters['A'])
            elif epoch < self.arg.freeze_graph_until:
                print('not now')
                print('continue')
            elif epoch > self.arg.freeze_graph_until:
                print('graph learn have been set')
                print('continue')

        self.model.train()  # 将模型设置为训练模式，只是改变模型的一些内置的参数运行方法，例如dropout
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)  # 直接调用实例对象，前提是实现了__call__方法
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):
        # 继承了父类，并在父类的基础上添加了新的参数
        # parameter priority: command line > config > default #命令行> 配置文件> 默认值
        # print('recognition')
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],  # 继承父类
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
