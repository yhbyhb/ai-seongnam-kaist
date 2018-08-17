"""
SUNGNAM-KAIST
2018/08/17
Authorized by SIIT, KAIST
Yekang Lee, Jaemyung Yu, and Junmo Kim
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf
import sys

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, lrn_rate, '
                     'weight_decay_rate,'
                     'relu_leakiness')

class SIITdcgan(object):

  def __init__(self, hps):

    self.hps = hps

    ## HYPERPARAMETER
    self.batch_size = hps.batch_size
    self.num_classes = 10
    self.z_dim = 100

    ## PLACEHOLDERS
    self.lrn_rate = tf.placeholder(tf.float32, shape=(), name='lrn_rate')
    self.z_noise = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
    self.label = tf.placeholder(tf.float32, [self.batch_size, self.num_classes])
    self.image = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1])

    self._extra_train_ops = []


  ## BUILD GRAPH
  def build_graph(self):
    ## GENERATE IMAGES FROM z(noise)
    with tf.variable_scope('generator') as scope:
      gen_image = self.generator(self.z_noise)
      self.gen_image = tf.nn.sigmoid(gen_image)

    ## DISCRIMINATOR (REAL? FAKE?)
    with tf.variable_scope('discriminator') as scope:
      real = self.discriminator(self.image)
      self.p_real = tf.nn.sigmoid(real)
      scope.reuse_variables()
      fake = self.discriminator(self.gen_image)
      self.p_fake = tf.nn.sigmoid(fake)

    ## DISCRIMINATOR LOSS
    cost_dis_real = self.sigmoid_loss(real, tf.ones_like(real))
    cost_dis_fake = self.sigmoid_loss(fake, tf.zeros_like(fake))
    self.cost_dis = cost_dis_real + cost_dis_fake

    ## GENERATOR LOSS
    self.cost_gen = self.sigmoid_loss(fake, tf.ones_like(fake))

    self.build_train_op()


  def build_train_op(self):
    optimizer = tf.train.AdamOptimizer(self.lrn_rate, beta1=0.5)
    dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    grads_dis = tf.gradients(self.cost_dis, dis_vars)
    grads_gen = tf.gradients(self.cost_gen, gen_vars)

    self.train_op_dis = optimizer.apply_gradients(zip(grads_dis, dis_vars),
                                         name='train_step_dis')
    self.train_op_gen = optimizer.apply_gradients(zip(grads_gen, gen_vars),
                                         name='train_step_gen')


  ## DISCRIMINATOR
  def discriminator(self, image):
    x = tf.reshape(image, [self.batch_size, 28, 28, 1])

    x = self._conv('conv1', x, 3, 1, 16, self._stride_arr(1))
    x = self._batch_norm('bn/conv1', x)
    x = self._relu(x)
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    x = self._conv('conv2',x, 3, 16, 32, self._stride_arr(1))
    x = self._batch_norm('bn/conv2', x)
    x = self._relu(x)
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    x = self._conv('conv3', x, 3, 32, 32, self._stride_arr(1))
    x = self._batch_norm('bn/conv3', x)
    x = self._relu(x)
    x = tf.reduce_mean(x, [1, 2])

    x = self._fully_connected('fc3', x, 1)
    return x


  ## GENERATOR
  def generator(self, z_noise):

    x = self._fully_connected('fc1', z_noise, 7*7*8)
    x = self._relu(x)
    x = tf.reshape(x, [-1, 7, 7, 8])

    x = self._conv('conv1',x, 3, 8, 16, self._stride_arr(1))
    x = self._batch_norm('bn/conv1', x)
    x = self._relu(x, 0.2)
    
    x = self._deconv('conv2', x, [self.batch_size, 14, 14, 32], 3, 3, 2, 2)
    x = self._batch_norm('bn/conv2', x)
    x = self._relu(x, 0.2)

    x = self._conv('conv3', x, 3, 32, 32, self._stride_arr(1))
    x = self._batch_norm('bn/conv3', x)
    x = self._relu(x, 0.2)
    
    x = self._deconv('conv4', x, [self.batch_size, 28, 28, 32], 3, 3, 2, 2)
    x = self._batch_norm('bn/conv4', x)
    x = self._relu(x, 0.2)
    
    x = self._conv('conv5', x, 3, 32, 1, self._stride_arr(1))
    
    return x    


  ## SIGMOID CROSS ENTROPY LOSS
  def sigmoid_loss(self, logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

  ## STRIDE 
  def _stride_arr(self, stride):
    return [1, stride, stride, 1]
  
  ## WEIGHT DECAY
  def decay(self):
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))

    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))
    
  ## CONVOLUTIONAL LAYER
  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      w = tf.get_variable(
          'weight/DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.uniform_unit_scaling_initializer(factor=2.0))

      y = tf.nn.conv2d(x, w, strides, padding='SAME')

    return y

  ## DECONVOLUTIONAL LAYER
  def _deconv(self, name, input_, output_shape,
         k_h=5, k_w=5, d_h=2, d_w=2):
    with tf.variable_scope(name):
      # filter : [height, width, output_channels, in_channels]
      w = tf.get_variable('DW', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.uniform_unit_scaling_initializer(factor=1))
    
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                  strides=[1, d_h, d_w, 1])

      biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
      deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

  ## RELU LAYER
  def _relu(self, x, leakiness=0.0):
    return tf.maximum(x, leakiness*x)

  ## FULLY CONNECTED LAYER
  def _fully_connected(self, name, x, out_dim, is_reuse=None):
    with tf.variable_scope(name, reuse=is_reuse):
      x = tf.reshape(x, [self.hps.batch_size, -1])
      w = tf.get_variable(
          'weight/DW', [x.get_shape()[1], out_dim],
          initializer=tf.uniform_unit_scaling_initializer(factor=2.0))
      b = tf.get_variable('bias/DW', [out_dim],trainable=False,
                          initializer=tf.constant_initializer())
      y = tf.nn.xw_plus_b(x, w, b)

    return y

  ## GLOBAL AVERAGE POOLING LAYER
  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

  ## BATCH NORMALIZATION LAYER
  def _batch_norm(self, name, x):
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y
      




