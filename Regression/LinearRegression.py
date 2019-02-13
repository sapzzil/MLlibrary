# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:05:30 2019

@author: sapzzil
"""


from CostFunction import CostFunction as cf
import tensorflow as tf

class LinearRegression():
    def __init__(self, cost_function = 'mse', optimizer = 'gradient descent', learning_rate = 0.1, epoch = 10000, name = 'Linear Regression'):
        self.cost_f_algorithm = cost_function
        self.optimizer_algorithm = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch         
        self.name = name
        
        
    def fit(self,x,y):   
        x_column = x.shape[1]
        y_column = y.shape[1]

        self.X = tf.placeholder(tf.float32, [None, x_column])
        self.Y = tf.placeholder(tf.float32, [None, y_column])
        self.model = self._create_model(x_column, y_column)
        
        cost_function = self._create_cost_function(self.cost_f_algorithm, self.model, self.Y)
        optimizer = self.setOptimizer(self.optimizer_algorithm, self.learning_rate)
        #optimizer = gd(self.learning_rate)
        train = optimizer.minimize(cost_function)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        for i in range(self.epoch):            
            _, grad = self.sess.run(train, feed_dict={self.X: x, self.Y: y})
            #print(self.sess.run(self.weight))
            #print(self.sess.run(self.bias))
            #print()
            if abs(grad) < 0.00001:
                break
            
        
    
    def predict(self,x):
        predicted_val = self.sess.run(self.model, feed_dict={self.X : x})
        return predicted_val

    def session_close(self):
        self.sess.close()
        
    def setOptimizer(self, optimizer_algorithm, learning_rate):
        if optimizer_algorithm == 'gradient descent':
            from Optimizer.GradientDescent import GradientDescent as gd
            return gd(learning_rate)
        else:
            print('no optimizer algorithm')
    
    
    def _create_model(self,x_column,y_column):     
         
        self.weight = self._create_weight(x_column, y_column)
        self.bias = self._create_bias(y_column)
        return tf.add(tf.matmul(self.X, self.weight), self.bias)
    
    def _create_weight(self, x_column, y_column):
        # x shape에서 weight 사이즈 계산해서 값 구함
        #w = tf.get_variable('weight', [x_column, y_column], initializer=tf.random_uniform_initializer(minval=-100.,maxval=100.))
        with tf.variable_scope(self.name):
            w = tf.Variable(tf.random_uniform([x_column, y_column], -100., 100.), name='weight')
        return w
    
    def _create_bias(self, y_column):
        #b = tf.get_variable('bias', [y_column], initializer=tf.random_uniform_initializer(minval=-100.,maxval=100.))
        with tf.variable_scope(self.name):
            b = tf.Variable(tf.random_uniform([y_column], -100., 100.), name= 'bias')
        return b
    
    def _create_cost_function(self, algorithm, model, y):
        cost_functions = {'mse' : cf.mse(model, y)}        
        return cost_functions[algorithm]
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass