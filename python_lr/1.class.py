
class Student(): #在python3中写不写object区别不大
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

    def avg(self,maths,chinese):
        self.maths=maths #这是参数，其实和属性没有多大区别，只不过不在实例化时创建，而是在调用具体方法时创建
        self.chinese=chinese
        mean_score=(maths+chinese)/2
        print(mean_score)

xiaoming=Student(name='xiaoming',sex='boy', age=18)

xiaoming.avg(100,50) #在执行这行语句前，没有maths和chinese这两个属性，执行后就有了

hk=Student(name='hk', sex='boy',age=23)
hk.print()
xiaoming.print()

print('start')
#step 1
print('\n\n')
hk.school='peking'
hk.print()
xiaoming.print() #此时xiaoming的学校仍然是uestc

#step 2
print('\n\n')
Student.school='shanghai'
hk.print()
xiaoming.print() #小明的学校变为shanghai，此时实例的类属性值是随着公有类属性值变化的

#step 3
xiaoming.school='chengdu'#小明的学校变为chengdu ，单独修改实例的类属性值,实际上这一步就是将公有类属性变成该实例的私有实例属性了
Student.school='dayi' #公有的类属性值变为dayi

