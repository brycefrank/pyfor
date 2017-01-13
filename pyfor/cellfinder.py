import random

def within_square(x,y, step, i):
    '''Determines if point 'i' is within a square of size 'step'''
    ix = i[0]
    iy = i[1]
    if (x<=ix and ix<x+step) and (y<=iy and iy<y+step):
        return True
    else:
        return False


def point_list(n):
    return [(random.uniform(-2,2), random.uniform(-2,2)) for i in range(n)]


mega_points = point_list(1000000)

for i in mega_points:
    print(within_square(0,0,1, i))

