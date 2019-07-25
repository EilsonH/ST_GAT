#Python中的构造函数是__init__函数。
# 在Python中，子类如果定义了构造函数，而没有调用父类的，那么Python不会自动调用，也就是说父类的构造函数不会执行

#文件夹中的__init__.py文件将该文件夹变成一个包

#调用父类的构造函数如下
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):
        print("person is talking....")


class Chinese(Person):

    def __init__(self, name, age, language):  # 先继承，在重构
        Person.__init__(self, name, age)  # 继承父类的构造方法，也可以写成：super(Chinese,self).__init__(name,age)
        self.language = language  # 定义类的本身属性

    def walk(self):
        print(self.name,'is walking...')

huangke=Chinese('hk',18,'mars')
huangke.talk()
huangke.walk()