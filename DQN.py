#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
class DQN:
    def __init__(self, stateSize, actionSize, learningRate, batchSize, name="DQNetwork"):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        
        self.actions = tf.Variable(np.zeros((batchSize, actionSize)), trainable=False, dtype=tf.float32)
        
        self.model = tf.keras.Sequential(name="dqn")
        
        self.model.add(tf.keras.layers.Conv2D(input_shape=(stateSize),
                                      filters=16,
                                      kernel_size=8,
                                      strides=4,
                                      padding="valid",
                                      activation="relu",
                                      kernel_initializer='glorot_uniform',
                                      name="conv1"))
        
        self.model.add(tf.keras.layers.Conv2D(
                                      filters=32,
                                      kernel_size=4,
                                      strides=2,
                                      padding="valid",
                                      activation="relu",
                                      kernel_initializer='glorot_uniform',
                                      name="conv2"))
        
        self.model.add(tf.keras.layers.Flatten())
        
        self.model.add(tf.keras.layers.Dense(
                                  units=256,
                                  activation ="relu",
                                  kernel_initializer = 'glorot_uniform',
                                  name="fc1"))
        
        self.model.add(tf.keras.layers.Dense(
                                      kernel_initializer='glorot_uniform',
                                      units=actionSize,
                                      activation=None,
                                      name="output"))
        
    def my_loss_fn(self, y_true, y_pred):
        self.Q = tf.reduce_sum(y_pred * self.actions)
        return tf.reduce_mean((y_true-self.Q)**2)
        
    def my_compile(self):
        self.model.compile(loss=self.my_loss_fn, optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learningRate))
        print("model compiled")
        
    def batch_loss(self, states, Qtarget, actionsMb):
        self.actions.assign(actionsMb)
        return self.model.train_on_batch(states, Qtarget)