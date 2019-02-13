# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:27:23 2019

@author: sapzzil
"""

import tensorflow as tf

class GradientDescent():
    def __init__(self, learning_rate= 0.1):
        self.lr = learning_rate
    
    def minimize(self, cost_function):
        # while 문에서 최소화
        update_list=[]
        for v in tf.trainable_variables():
            #print(tf.gradients(cost_function, v, stop_gradients= v))
            #print(tf.multiply(self.lr, tf.gradients(cost_function, v, stop_gradients= v)))
            grad = tf.gradients(cost_function, v, stop_gradients= v)[0]
            update_val = v.assign_sub(tf.multiply(self.lr, grad))
            update_list.append(update_val)
            #updated_v = tf.subtract(v , tf.multiply(self.lr, tf.gradients(cost_function, v, stop_gradients= v)))
            #v.assign(updated_v)               

        return update_list, grad
        
        
    
    
    
if __name__ == "__main__":
    import numpy as np
    x_data = np.array(range(100))
    x_data = x_data/ max(x_data)
    y_data = np.array(range(100))
    y_data = y_data /y_data.max()
    w = tf.get_variable('weight', [1], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('bias', [1], initializer=tf.truncated_normal_initializer())
    y = tf.placeholder(dtype='float')
   # print(tf.trainable_variables())
    x = tf.placeholder(dtype='float')
    
        
    #print(tf.trainable_variables())
    model = tf.add(tf.multiply(w,x),b)
    from CostFunction import CostFunction as cf
    loss = cf.mse(model, y)
    optmz = GradientDescent()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:        
        sess.run(init)
        
        for i in range(200):
            _, grad = sess.run(optmz.minimize(loss) ,feed_dict={x : x_data, y : y_data})
            if np.abs(grad) < 0.0001 :
                break
            weight = sess.run(w)
            bias = sess.run(b)
            
            print(grad)
            print(weight)
            print(bias)
            print()

