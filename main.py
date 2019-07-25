# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import argparse
import sys
import os
# torchlight
import torchlight #轻量版本的torch
from torchlight import import_class

#python main.py recognition -c config/st_gcn/kinetics-skeleton/test.yaml
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # 创建一个解析器，ArgumentParser实例
    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict() #定义了一个字典对象
    #字典对象中的每个元组，包含一个key 和 value，即：
    #key:value
    # 构建一个字典，用不同的键来表示不同的值
    processors['recognition'] = import_class('processor.recognition.REC_Processor') # import_class意思就是从指定路径中导入一个类
    processors['demo'] = import_class('processor.demo.Demo')
    #endregion yapf: enable

    # add sub-parser
    # arg0 = parser.parse_args() #解析命令行参数
    # 构建一个子命令解析器，_SubParsersAction，注意这行语句不会改变parser本身的值
    # 并且parser只能调用一次add_subparsers方法，也就说只能有一个subparser
    #subparser用于不同子命令的处理，是一个从parser对象创建的特殊动作对象，该对象只有一个add_parser方法

    #这里的dest和之前argumentparser的dest参数不同，这里是指子命令保存到的属性名
    #这里processor是个属性变量，其中保存的就是子命令的识别名称，也就是recognition和demo
    subparsers = parser.add_subparsers(dest='processor' )##如果没有dest='processsor'，那么在arg中就不会有processor='recognition',这个操作实际上就是为
    print(subparsers)

    for k, p in processors.items(): #.items()方法返回可遍历的(键, 值) 元组数组，k是键，p是值
        #parents即一个列表，列表中的所有对象变量都要被包含在内
        #p.get_parser将class类型的p变成一个和parser相同的ArgumentParser实例
        print(k)
        print(p)
        subparsers.add_parser(k, parents=[p.get_parser()])  #p就是上面的那个字符串，这里是调用p的get_parser()方法，返回一个argumentparser对象
        # print(subparsers)
    #这样就得到了两个子命令
    # read arguments
    #从recognition是用于命令区分的，其他的全部
    #'recognition -c config/st_gcn/kinetics-skeleton/test.yaml'.split()
    arg = parser.parse_args() #解析命令行参数

    # start
    Processor = processors[arg.processor] #arg.processor=recognition
    #那么这里processor=processors['recognition']，也就是待导入的那个类

    # 如果不是通过命令行调试或者不是模拟命令调试，那么sys.argv就只有一个文件路径名称,就不能作为参数
    # 并且之后调用的包中的sys.argv也无法获取到参数，因此要么直接赋值，
    print(sys.argv[2:])
    # 实例化类 REC_processor 得到p，并传入参数sys.argv[2:]，这个参数就是三个文件中的argv
    p = Processor(sys.argv[2:])  #在实例化的时候就要运行一次构造函数

    print("final stage")
    p.start() #start方法定义在processor中，在recognition中继承
