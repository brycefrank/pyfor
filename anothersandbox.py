from multiprocessing import Pool
import random

list1 = [int(10000000*random.random()) for i in range(100000000)]
def f(x):
    return x*x

if __name__ =='__main__':
    p = Pool(2)
    print(p.map(f,list1))

