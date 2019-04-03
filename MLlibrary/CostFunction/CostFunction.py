# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:47:48 2019

@author: sapzzil
"""
import tensorflow as tf

def mse(model,y):
    return tf.reduce_mean(tf.square(model-y))

def cross_entropy(model,y):
    return tf.reduce_mean(-y * tf.log(model) - (1-y)* tf.log(1-model))
    