

list2 = [10,20,30,40,50]



def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def thing(n):
    if n == 0:

        return 1
    else:
        return n - thing(n-1)

def list_append(n, list1, start="end"):
    step = list1[1] - list1[0]
    if start == "end":
        last = list1[-1]
        for i in range(n):
            list1.append(last + (i+1)*step)
        return list1
    if start == "beginning":
        first = list1[0]
        for i in range(n):
            list1.insert(0, first - (i+1)*step)
        return list1

print(list_append(3, list2, start="beginning"))