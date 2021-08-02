from multiprocessing import Process


def fun1(name, i):
    print('测试%s %d多进程' % (name, i))


# if __name__ == '__main__':
#     process_list = []
#     for i in range(20):
#         # 开启5个子进程执行fun1函数
#         p = Process(target=fun1, args=('Python', i))
#         # 实例化进程对象
#         p.start()
#         process_list.append(p)

#     for i in process_list:
#         p.join()

#     print('结束测试')


from multiprocessing import  Process
def test(a,b,c):
    print(a,b,c)
class MyProcess(Process): #继承Process类
    def __init__(self,index,a,b,c):
        # super(MyProcess,self).__init__()
        super().__init__()
        self.index = index
        self.a = a
        self.b = b
        self.c = c
    # @staticmethod
    # def test(a,b,c):
    #     print(a,b,c)
    def run(self):
        print('测试%s多进程' % self.index)
        test(self.a,self.b,self.c)


if __name__ == '__main__':
    process_list = []
    for i in range(5):  #开启5个子进程执行fun1函数
        p = MyProcess(i,i+1,i+2,i+3) #实例化进程对象
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('结束测试')

