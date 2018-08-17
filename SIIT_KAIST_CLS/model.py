"""
SEUNGNAM-KAIST
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
                     'weight_decay_rate')


class Deepnet(object):

  def __init__(self, hps, mode):

    self.hps = hps

    ## HYPERPARAMETER
    self.total_classes = 10
    self.relu_leakiness = 0
    self.mode = mode
    self._extra_train_ops = []
    self.num_residual_units = 3

    ## PLACEHOLDERS
    self.lrn_rate = tf.placeholder(tf.float32, shape=(), name='lrn_rate')
    self.images = tf.placeholder(tf.float32, shape=(hps.batch_size, 28, 28, 1), name='images')
    self.labels = tf.placeholder(tf.float32, shape=(hps.batch_size, self.total_classes), name='labels')


  ## BUILD GRAPH
  ## Option 1: resnet
  ## Option 2: vggnet
  ## Option 3: simple network (fill in the blank!)
  ## Option 4: DIY network (fill in the blank!)

  def build_graph(self):
    ## GLOBAL ITERATION
    self.global_step = tf.contrib.framework.get_or_create_global_step()

    ## FEATURE EXTRACTION
    with tf.variable_scope('embed') as scope:
      #feats = self.resnet(self.images)
      feats = self.vggnet(self.images)
      #feats = self.simple_network(self.images)
      #feats = self.DIY_network(self.images)

    ## LOGITS
    logits = self._fully_connected('logits', feats, self.total_classes)

    ## SOFTMAX 
    self.prediction = tf.nn.softmax(logits)

    ## COST FUNCTION (CROSS ENTROPY)
    cent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
    self.cost_cls = tf.reduce_mean(cent, name='cent')

    self.build_cost()

    if self.mode == 'train':
      self.build_train_op()


  def build_cost(self):
    if self.mode == 'train':
      self.cost = self.cost_cls
      self.cost += self.decay()
    elif self.mode == 'test':
      self.cost = self.cost_cls

  ## OPTIMIZER (SGD+MOMENTUM)
  def build_train_op(self):
    optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)
    train_vars = tf.trainable_variables()
    grads = tf.gradients(self.cost, train_vars)
    apply_op = optimizer.apply_gradients(zip(grads, train_vars),
                                         global_step=self.global_step,
                                         name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  ## L2 WEIGHT DECAY
  def decay(self):
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))

    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  ################################
  ####### FILL IN THE BLANK ######
  ################################
  ################################
  ####### BE CREATIVE! ###########
  ################################
  def DIY_network(self, images):
    """ Your Code """















    img_feat = self._global_avg_pool(x)
    return img_feat



  ################################
  ####### FILL IN THE BLANK ######
  ################################
  def simple_network(self, images):
    """ Your Code """




















    img_feat = self._global_avg_pool(x)
    return img_feat


  ## VGGNET
  ## Very Deep Convolutional Networks for Large-Scale Image Recognition
  ## https://arxiv.org/abs/1409.1556

  def vggnet(self, images, labels=None):
    x = self._conv('conv1', images, 7, 1, 16, self._stride_arr(1))
    x = self._batch_norm('bn1', x)
    x = self._relu(x, self.relu_leakiness)
    x = self._conv('conv2', x, 3, 16, 16, self._stride_arr(1))
    x = self._batch_norm('bn2', x)
    x = self._relu(x, self.relu_leakiness)
    x = tf.nn.avg_pool(x, self._stride_arr(2), self._stride_arr(2), 'VALID')

    x = self._conv('conv3', x, 3, 16, 32, self._stride_arr(1))
    x = self._batch_norm('bn3', x)
    x = self._relu(x, self.relu_leakiness)
    x = self._conv('conv4', x, 3, 32, 32, self._stride_arr(1))
    x = self._batch_norm('bn4', x)
    x = self._relu(x, self.relu_leakiness)
    x = tf.nn.avg_pool(x, self._stride_arr(2), self._stride_arr(2), 'VALID')

    x = self._conv('conv5', x, 3, 32, 64, self._stride_arr(1))
    x = self._batch_norm('bn5', x)
    x = self._relu(x, self.relu_leakiness)
    x = self._conv('conv6', x, 3, 64, 64, self._stride_arr(1))
    x = self._batch_norm('bn6', x)
    x = self._relu(x, self.relu_leakiness)
    img_feat = self._global_avg_pool(x)
    return img_feat


  ## RESNET
  ## Deep Residual Learning for Image Recognition
  ## https://arxiv.org/abs/1512.03385
    
  def resnet(self, images, labels=None):

    with tf.variable_scope('init'):
      x = self._conv('conv1', images, 3, 1, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self.residual
    filters = [16, 16, 32, 64]

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, self.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, self.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, self.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
    
    with tf.variable_scope('unit_last'):
      x = self._batch_norm('bn_last', x)
      x = self._relu(x, self.relu_leakiness)
      img_feat = self._global_avg_pool(x)
      return img_feat


  def residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('bn_init', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('bn_init', x)
        x = self._relu(x, self.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    return x

  ## STRIDE
  def _stride_arr(self, stride):
    return [1, stride, stride, 1]
  
  ## CONVOLUTIONAL LAYER
  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      w = tf.get_variable(
          'weight/DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.uniform_unit_scaling_initializer(factor=2.0))

      y = tf.nn.conv2d(x, w, strides, padding='SAME')

    return y

  ## RELU LAYER
  def _relu(self, x, leakiness=0.0):
    return tf.maximum(x, leakiness*x)

  ## FULLY-CONNECTED LAYER
  def _fully_connected(self, name, x, out_dim, is_reuse=None):
    with tf.variable_scope(name, reuse=is_reuse):
      x = tf.reshape(x, [self.hps.batch_size, -1])
      w = tf.get_variable(
          'weight/DW', [x.get_shape()[1], out_dim],
          initializer=tf.uniform_unit_scaling_initializer(factor=2.0))
      b = tf.get_variable('bias/DW', [out_dim],
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

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

