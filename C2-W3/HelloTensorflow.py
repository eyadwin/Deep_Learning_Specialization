import numpy as np
import tensorflow as tf

#GradientTape is used in eager execution so you should add tf.enable_eager_execution() at the beginning of your program.
#tf.enable_eager_execution()

#minimize the function  w^2 - 10w +25, the minimum is 5

w = tf.Variable(0,dtype=tf.float32)#define the parameter w, variable is intitialized to zero
optimizer = tf.keras.optimizers.Adam(0.1)#define Adam as the optimization algorithm we will use, and learning rate is 0.1


#in tensorflow only forawrd prop, and the tensorflow will automatically do the backprop
#The intuition behind the name gradient tape is by an analogy to the old-school cassette tapes,
# where Gradient Tape will record the sequence of operations as you're computing the cost function
# in the forward prop step. Then when you play the tape backwards, in backwards order, it can revisit
# the order of operations in reverse order, and along the way, compute backprop and the gradients.
def train_step():
    with tf.GradientTape() as tape:
        cost = w**2 - 10*w +25
    trainable_variables =[w]#list with only w
    grads = tape.gradient(cost,trainable_variables)#compute gradients
    optimizer.apply_gradients(zip(grads,trainable_variables))    #use optimizer to apply gradients, zip function take the two lists takes two lists and pairs up the corresponding elements.

for i in range (1000):
    train_step()
print(w)
