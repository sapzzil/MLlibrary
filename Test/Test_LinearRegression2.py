# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:23:16 2019

@author: sapzzil
"""

from Regression.LinearRegression import LinearRegression as lr
import numpy as np

x_data2 = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
x_data2 = np.asarray(x_data2)
x_data2 = x_data2 /20 
y_data2 = [1,2,3,4]
y_data2 = np.asarray(y_data2).reshape(-1,1) 
y_data2 = y_data2/ 20
test_x2 = [[5,6,7],[9,10,11]]
test_x2 = np.asarray(test_x2)
test_x2 = test_x2 /20
model2 = lr(cost_function = 'mse', optimizer = 'gradient descent', epoch = 5000, name= 'test2')
model2.fit(x_data2, y_data2)
predicted_val2 = model2.predict(test_x2)
model2.session_close()
print('################# My MLlibrary #######################')
print(' input values \n {} \n output values \n {}'.format(test_x2*20, predicted_val2*20))

import tensorflow as tf

W2 = tf.Variable(tf.random_uniform([3,1], -100., 100.))
b2 = tf.Variable(tf.random_uniform([1], -100., 100.))

X2 = tf.placeholder(tf.float32)
Y2 = tf.placeholder(tf.float32)

hypothesis2 =  tf.matmul(X2 , W2) + b2

cost2 = tf.reduce_mean(tf.square(hypothesis2 - Y2))

rate2 = tf.Variable(0.1)
optimizer2 = tf.train.GradientDescentOptimizer(rate2)
train2 = optimizer2.minimize(cost2)

init2 = tf.global_variables_initializer()

sess2 = tf.Session()
sess2.run(init2)

for step in range(5000):
    sess2.run(train2, feed_dict={X2: x_data2, Y2: y_data2})
    if step % 20 == 0:
        step, sess2.run(cost2, feed_dict={X2: x_data2, Y2: y_data2})

tensorflow_predicted_val2 = sess2.run(hypothesis2, feed_dict={X2: test_x2})
sess2.close()
print('################# Tensorflow MLlibrary #######################')
print('input values : {}\noutput values : {}'.format(test_x2*20, tensorflow_predicted_val2*20))