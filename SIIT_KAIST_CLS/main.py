
"""
SEUNGNAM-KAIST
2018/08/17
Authorized by SIIT, KAIST
Yekang Lee, Jaemyung Yu, and Junmo Kim
"""

from __future__ import print_function
import sys
import time
from datetime import datetime
from random import randint
#from scipy.misc import imsave
#from scipy.misc import imread

from glob import glob
from util import *

import numpy as np
import model as Deepnet
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_gpus', '1',
                           'cpu 0 / gpu 1.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('log_dir', 'logs/',
                           'Directory to logs.')
tf.app.flags.DEFINE_integer('batch_size', '100',
                           'batch_size.')
tf.app.flags.DEFINE_float('base_lr', '0.1',
                           'base learning rate.')


def train(hps):

  ## NUM_DATA
  NUM_DATA = 1000

  ## READ MNIST INPUTS
  mnist = input_data.read_data_sets('./data/', one_hot=True)

  ## TRAIN/VAL SPLITS
  train_images = mnist.train.images
  train_labels = mnist.train.labels
  train_images = train_images.reshape([-1, 28, 28, 1])
  train_images = train_images[0:NUM_DATA]
  train_labels = train_labels[0:NUM_DATA]
  val_images = mnist.validation.images
  val_labels = mnist.validation.labels
  val_images = val_images.reshape([-1, 28, 28, 1])

  ## RANDOM SHUFFLING
  order = np.random.permutation(train_images.shape[0])
  order = order.astype(np.int32)
  train_images = train_images[order, :, :, :]
  train_labels = train_labels[order, :]

  ## BUILD GRAPH
  model = Deepnet.Deepnet(hps, FLAGS.mode)
  model.build_graph()

  ## MAKE SESSION
  saver = tf.train.Saver()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True # Use memory as much as needed 
  sess = tf.InteractiveSession(config=config)

  ## INITIALIZE SESSION
  sess.run(tf.global_variables_initializer())

  ## HYPERPARAMETER
  lrn_rate = hps.lrn_rate
  max_step = 5000

  for step in range(1, max_step+1):

    ## STEP DECAYING
    if step < 2000:
      lrn_rate = hps.lrn_rate
    elif step < 4000:
      lrn_rate = 0.1 * hps.lrn_rate
    else:
      lrn_rate = 0.01 * hps.lrn_rate

    ## BATCH SELECTION
    k_start = hps.batch_size * step % train_images.shape[0]
    k_end = hps.batch_size * step % train_images.shape[0] + hps.batch_size
    batch_images = train_images[k_start:k_end, :, :, :]
    batch_labels = train_labels[k_start:k_end, :]

    ## RUN SESSION
    start_time = time.time()
    (_, loss, loss_cls, truth, prediction, train_step) = sess.run(
        [model.train_op, model.cost, model.cost_cls, model.labels, model.prediction, 
         model.global_step],
         feed_dict={model.lrn_rate: lrn_rate,
				  					model.images: batch_images,
                    model.labels: batch_labels})
    duration = time.time() - start_time
    sec_per_batch = float(duration)

    ## CALCULATE ACCURACY
    truth = np.argmax(truth, axis=1)
    prediction = np.argmax(prediction, axis=1)
    precision = np.mean(truth == prediction)
    if step % 5 == 0:
      print('(TRAINING) Iteration %d, Lr: %.3f, Loss: %.4f, Acc: %.4f (duration: %.3fs)' 
                  % (step, lrn_rate, loss_cls, precision, sec_per_batch))
    if step % 100 == 0:
      save_path = saver.save(sess, FLAGS.log_dir + "iter_%d" % (step))
      print('Model saved in file: %s.' % save_path)

    ## VALIDATION
    if step % 100 == 0:
      total_prediction = 0
      correct_prediction = 0
      for step in range(100):
        ## BATCH SELECTION
        k_start = hps.batch_size * step % val_images.shape[0]
        k_end = hps.batch_size * step % val_images.shape[0] + hps.batch_size
        batch_images = val_images[k_start:k_end, :, :, :]
        batch_labels = val_labels[k_start:k_end, :]
        (loss, truth, prediction) = sess.run(
            [model.cost_cls, model.labels, model.prediction],
            feed_dict={model.images: batch_images,
                       model.labels: batch_labels})

        truth = np.argmax(truth, axis=1)
        prediction = np.argmax(prediction, axis=1)
        precision = np.sum(truth == prediction)
        total_prediction += prediction.shape[0]
        correct_prediction += precision
  
      precision = 1.0 * correct_prediction / total_prediction
      print('(VALIDATION) ACC: %.3f' % precision)

  print('Optimization done.')
  print('Save the checkpoint.')



def test(hps):

  ## READ MNIST INPUTS
  mnist = input_data.read_data_sets('./data/', one_hot=True)
  test_images = mnist.test.images
  test_labels = mnist.test.labels
  test_images = test_images.reshape([-1, 28, 28, 1])

  ## BUILD GRAPH
  model = Deepnet.Deepnet(hps, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()

  ## MAKE SESSION 
  ## if there exist checkpoints, then restore it.
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_dir)
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    print('No model to eval yet at %s' % FLAGS.log_dir)
    return
  print('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)

  ## HYPERPARAMETER
  batch_size = 100
  total_prediction = 0
  correct_prediction = 0

  for step in range(100):

    ## BATCH SELECTION
    k_start = hps.batch_size * step % test_images.shape[0]
    k_end = hps.batch_size * step % test_images.shape[0] + hps.batch_size
    batch_images = test_images[k_start:k_end, :, :, :]
    batch_labels = test_labels[k_start:k_end, :]

    ### RUN SESSION
    start_time = time.time()
    (loss, truth, prediction) = sess.run(
        [model.cost, model.labels, model.prediction],
        feed_dict={model.images: batch_images,
                    model.labels: batch_labels})

    truth = np.argmax(truth, axis=1)
    prediction = np.argmax(prediction, axis=1)
    precision = np.sum(truth == prediction)
    print('Loss: %.4f, Acc: %.4f' 
               % (loss, precision))
    total_prediction += prediction.shape[0]
    correct_prediction += precision
  
  precision = 1.0 * correct_prediction / total_prediction
  print('Acc: %.3f' % precision)


## MAIN FUNCTION
def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  hps = Deepnet.HParams(batch_size=FLAGS.batch_size,
                           lrn_rate=FLAGS.base_lr,
                           weight_decay_rate=0.0001)

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'test':
      test(hps)

if __name__ == '__main__':
  tf.app.run()

