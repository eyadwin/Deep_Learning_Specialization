#Vectorization example
#Compare the time it take to multiply two vectors
#using vectorized version (without loop) and non-vectorized version (i.e., for loop)

import numpy as np
a=np.array([1,2,3,4])
print(a)

#lets calculate time
import time
a=np.random.rand(1000000) #create 1 random million value
b=np.random.rand(1000000)

tic = time.time()
c= np.dot(a,b)
toc = time.time()

print(c)
print("Vectorized version:" + str(1000*(toc-tic)) + "ms")

#non victorized version
c=0
tic = time.time()
for i in range(1000000):
    c+=a[i]*b[i]

toc = time.time()
print(c)

print("For loop:" + str(1000*(toc-tic)) + "ms")

#Broadcasting example
A = np.array([[56.0,0.0,4.4,68.0],
             [1.2,104.0,52.0,8.0],
             [1.8,135.0,99.0,0.9]])
print (A)

cal = A.sum(axis=0) # sum values in matrix vertically (by column)
print("cal")
print(cal)

print("percentage")
percentage = 100 * A/cal.reshape(1,4)
print(percentage)
