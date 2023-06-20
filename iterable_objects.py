# 迭代器定义: 如果一个类中有__iter__()方法且返回一个迭代器对象(生成器是特殊的迭代器), 则我们称这个类创建的对象叫可迭代对象。

# 例子如下：
class Foo(object): 

    def __iter__(self):
        return "迭代器对象(生成器对象)"
    
obj = Foo()
for item in obj:    # item若能调用其__iter__返回迭代器对象，则item就是可迭代对象
    pass