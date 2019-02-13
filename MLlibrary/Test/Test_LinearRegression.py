# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:28:53 2019

@author: sapzzil
"""


from Regression.LinearRegression import LinearRegression as lr
import numpy as np

# X,Y, test_x 변수
x_data = np.array(range(100)).reshape(-1,1)
x_data = x_data/ max(x_data)
y_data = np.array(range(100)).reshape(-1,1)
y_data = y_data /y_data.max()

# 모델 생성
model = lr(cost_function = 'mse', optimizer = 'gradient descent', epoch = 2001, name = 'test1')

# 모델 cost function 구하기
    # 여기선 mse(mean squared error)
# model.cost_function('mse')
# model.optimizer('gradient descent')

# 모델 XY 넣고
# 모델 값 세팅
# 모델 피팅(트레인)
model.fit(x_data, y_data)


test_x = np.array(range(101,105)).reshape(-1,1)
# 예측값
predicted_val = model.predict(test_x)
model.session_close()
print('################# My MLlibrary #######################')
print('input values : {}\noutput values : {}'.format(test_x, predicted_val))



import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -100., 100.))
b = tf.Variable(tf.random_uniform([1], -100., 100.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        step, sess.run(cost, feed_dict={X: x_data, Y: y_data})

tensorflow_predicted_val = sess.run(hypothesis, feed_dict={X: test_x})
sess.close()
print('################# Tensorflow MLlibrary #######################')
print('input values : {}\noutput values : {}'.format(test_x, tensorflow_predicted_val))

