from time import time

from driver import Driver

t = time()
print("started at:", t)
Driver().drive()
t2 = time()
print("ended at:", t2)
print("total duration: ", t2 - t)

