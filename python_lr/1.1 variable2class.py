#如果将外部的变量作为类参数，在初始化时作为类属性参数传入，那么传入的是地址还是？

class Student():
    school='uestc'

    # 初始化方法，在实例化时默认执行一次
    # 通常写到初始化方法中的是属性
    def __init__(self,name,sex,age):
        self.name=name
        self.sex=sex
        self.age=age

    def print(self):
        print('类属性值',Student.school)
        print('该实例的类属性值',self.school)
        print('该实例的属性值',self.name)
        print('年龄',self.age)

    def avg(self,maths,chinese):
        self.maths=maths #这是参数，其实和属性没有多大区别，只不过不在实例化时创建，而是在调用具体方法时创建
        self.chinese=chinese
        mean_score=(maths+chinese)/2
        print(mean_score)

age=18
hk=Student('HK','boy',age)
yj=Student('yj','girl',age)
xm=Student('xm','boy',age)

print(hk.age)
print(yj.age)
print(xm.age)
yj.age=20
print(xm.age)