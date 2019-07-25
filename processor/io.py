#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn

# torchlight
#这是一个很少有人使用的模块
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

class IO():
    """
        IO Processor
    """

    def __init__(self, argv=None):#这里argv=None是默认值,并且这个argv是作为参数从外部传入，而不是sys.argv直接从命令行获取
        #argv的值就是['-c', 'config/st_gcn/kinetics-skeleton/train.yaml'] 也就是配置文件路径
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()

    def load_arg(self, argv=None): #argv同init里面的argv
        # get_parser是一个自定义的静态方法（函数），虽然不以self作为参数，但是self可以调用
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv) #用这个定义的解析器去解析argv，得到的就是与训练有关的默认值，注意p不仅包含argv，还包含默认值
        if p.config is not None: #如果输入了yaml文件路径，就不为空
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)

            # update parser from config file
            #p是一个namespace对象，vars(p)将其变成字典的形式,keys返回字典的所有键值
            key = vars(p).keys()
            #如果配置文件中存在默认键中没有的值，就报错
            for k in default_arg.keys(): #配置文件中的键值
                if k not in key:  #默认的键值
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg) #从配置文件获取的值作为默认值，以字典形式

        self.arg = parser.parse_args(argv) #现在parser由默认值，和config值构成，全部作为self的一个arg属性
        ###
        print(self.arg.model) #net.st_gcn.Model

    def init_environment(self):
        self.io = torchlight.IO( #class 'torchlight.io.IO',这里的torchlight.IO就是一个容器
            self.arg.work_dir, #存储结果的路径，默认为'./work_dir/tmp'
            save_log=self.arg.save_log, #是否保存日志，默认true
            print_log=self.arg.print_log) #是否打印日志，默认true
        self.io.save_arg(self.arg)

        # gpu
        if self.arg.use_gpu: #如果指定了使用GPU，就默认使用0号GPU，否则使用CPU
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights,
                                              self.arg.ignore_weights)

    def gpu(self):
        # move modules to gpu
        self.model = self.model.to(self.dev) #将模式切换为GPU模式
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

        # model parallel
        #如果长度大于1，则并行计算
        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

    @staticmethod           #这是一个静态方法，不以self作为参数，所以不能调用类变量和实例变量
                            ###相应的还有@classmethod 类方法，只能访问类变量，不能访问实例变量
    def get_parser(add_help=False):#
        # print('father parser')
        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='IO Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file') #通向yaml文件的

        # processor
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        #endregion yapf: enable

        return parser
