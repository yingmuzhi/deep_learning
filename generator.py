import sys, os
sys.path.append(os.path.dirname(__file__))
from timer import Timer

# list 会 一次性将数据全部存储进内存; 而 generator 不会一股脑将数据存进内存，而是根据算法一次一次生成。
timer2 = Timer()
diy_list = [i for i in range(7200)]         # list
timer2_1 = timer2.stop()

timer2.start()
diy_generator = (i for i in range(7200))    # generator
timer2_2 = timer2.stop()
print(timer2.times)         # 当数据量为7200000个时，[3.2351293563842773, 9.5367431640625e-06] 可以看出列表生成器更快   | 类似于生成len=7200000的list
print(type(diy_generator))  # <class 'generator'>

# 比价时间
timer = Timer()
for i in diy_list:
    # print(i)
    pass
time1 = timer.stop()

timer.start()
for i in diy_generator:
    # print(i)
    pass
time2 = timer.stop()
print(time1, time2, timer.times)    # 当数据量为7200000个时，使用 for in 迭代输出，输出时间为 [2.6780059337615967, 7.6640541553497314] 可以看到列表输出更快 | 类似于batch_size = 7200000 

# 总结，当你数据量很大，如一百万张图片左右（经验值）的时候，推荐使用IterDataset。
# 当你在__init__()方法中生成data_path时间 大于 调用batch_size个 __getitem__()方法时候，可以考虑使用iterDataset


# 如果一个函数里面被定义了 yield，那么这个函数就是一个生成器
def foo():
    print("1")
    yield
    print("2")
    yield
    print("3")
    yield

f = foo()   # f 是生成器对象（内部是根据生成器类generator创建的对象，生成器类也声明了：__iter__(), __next__()方法。生成器属于一种特殊的迭代器）。
print(type(f))
next(f)
next(f)
# next(f)
next(f)
# yield在函数中用于将函数变成迭代器。你可以使用python的built-in function(next)来调用重写的magic method(__next__())
# 当被next()调用的时候，yield的作用类似于return，只不过他会从上一次yield结束的地方开始运行，直到下一次yield的地方。当你越界后，会报错StopIteration。


# 常见yield函数使用yield返回值，如下：
def foo2():
    for i in range(10):
        yield i * 2
for i in foo2():    # 当对可迭代对象（如生成器就是可迭代对象）for循环时候，本质上是先调用可迭代对象的__iter__()方法得到迭代器对象，在内部执行迭代器对象的__next__()方法。
    print(i)