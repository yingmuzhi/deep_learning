'''
迭代器类型的定义：
    1. 类中定义__iter__和__next__方法
    2. __iter__方法返回对象本身, 即: self
    3. __next__方法, 返回下一个数据, 如果没有数据则StopIteration
'''

# 创建一个迭代器对象
class EvenNumbers:
    def __init__(self, max_number):
        self.max_number = max_number
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max_number:
            raise StopIteration
        result = self.current
        self.current += 2
        return result

even_numbers = EvenNumbers(10)
print(type(even_numbers))   # <class '__main__.EvenNumbers'>

print(next(even_numbers))
print(next(even_numbers))

for _, number in enumerate(even_numbers):
    print("in", number)
# # the same as below:
# for number in even_numbers:
#     print(number) 
