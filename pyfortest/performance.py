# This tests the performance of different functions on a small data set.

import time
import pyfor

def time_func(func_string):
    start = time.time()
    eval(func_string)
    end = time.time()
    return(end - start)


## Initialize cloud
#print('Initialize cloud: {}'.format(time_func('pyfor.cloud.Cloud("data/test.las")')))


#test_cloud = pyfor.cloud.Cloud("data/test.las")

## Initialize grid from cloud with different sizes
#print('Initialize grid - 0.25: {}'.format(time_func('test_cloud.grid(0.25)')))
#print('Initialize grid - 0.5: {}'.format(time_func('test_cloud.grid(0.5)')))
#print('Initialize grid - 1: {}'.format(time_func('test_cloud.grid(1)')))
#print('Initialize grid - 10: {}'.format(time_func('test_cloud.grid(10)')))


## Normalize grid at different sizes
#print('Normalize grid - 0.25: {}'.format(time_func('test_cloud.normalize(0.25)')))
#print('Normalize grid - 0.5: {}'.format(time_func('test_cloud.normalize(0.5)')))
#print('Normalize grid - 1: {}'.format(time_func('test_cloud.normalize(1)')))

