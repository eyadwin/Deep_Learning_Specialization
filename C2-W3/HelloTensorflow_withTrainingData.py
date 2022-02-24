import numpy as np
import tensorflow as tf

#used in eager execution so you should add tf.enable_eager_execution() at the beginning of your program.
#tf.enable_eager_execution()

#minimize the function  w^2 - 10w +25, the minimum is 5

w = tf.Variable(0,dtype=tf.float32)#define the parameter w, variable is intitialized to zero
x= np.array([1.0,-10.0,25.0],dtype=np.float32)#list coefiecient of cost function
optimizer = tf.keras.optimizers.Adam(0.1)#define Adam as the optimization algorithm we will use, and learning rate is 0.1

#tensorflow automatically calculate derviatives and knows how to optimize the cost
#it constrcut a computation graph
#You just do the forward prop, and tensorflow automically figure out the backward calculations
def training(x,w,optimizer):
    def cost_fn():
        return x[0]* w**2 + x[1]*w +x[2]
    for i in range(1000):
        optimizer.minimize(cost_fn,[w])
    return w

w = training(x, w, optimizer)
print(w)
