"""
SUNGNAM-KAIST
2018/08/17
Authorized by SIIT, KAIST
Yekang Lee, Jaemyung Yu, and Junmo Kim
"""

import sys
import os
import numpy as np
import tensorflow as tf
import model as net
from tensorflow.examples.tutorials.mnist import input_data
from util import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_gpus', '1',
                            'cpu 0 / gpu 1.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('log_dir', 'logs/',
                            'Directory to logs.')
tf.app.flags.DEFINE_integer('batch_size', '100',
                           'batch_size.')
tf.app.flags.DEFINE_float('base_lr', '0.001',
                           'base learning rate.')


def train(hps):

  ## NUM DATA
  NUM_DATA = 50000

  ## READ MNIST INPUTS
  mnist = input_data.read_data_sets('./data/', one_hot=True)
  train_images = mnist.train.images
  train_labels = mnist.train.labels
  train_images = train_images.reshape([-1, 28, 28, 1])
  train_images = train_images[0:NUM_DATA]
  train_labels = train_labels[0:NUM_DATA]

  val_images = mnist.validation.images
  val_labels = mnist.validation.labels
  val_images = val_images.reshape([-1, 28, 28, 1])

  ## BUILD GRAPH
  model = net.SIITdcgan(hps)
  model.build_graph()

  ## MAKE SESSION
  saver = tf.train.Saver()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True # Use memory as much as needed 
  sess = tf.InteractiveSession(config=config)

  ## INITIALIZATION
  sess.run(tf.global_variables_initializer())

  ## HYPERPARAMETERS
  z_dim = 100
  num_classes = 10
  max_step = 5000
  lrn_rate = hps.lrn_rate

  for step in range(1, max_step+1):

    ## BATCH SELECTION
    k_start = hps.batch_size * step % train_images.shape[0]
    k_end = hps.batch_size * step % train_images.shape[0] + hps.batch_size
    batch_images = train_images[k_start:k_end, :, :, :]
    batch_labels = train_labels[k_start:k_end, :]
    z_noise = np.random.uniform(-1, 1, size=[hps.batch_size, z_dim]).astype(np.float32)

    ## RUN SESSION
    ## 1. UPDATE GENERATOR
    _, gen_loss = sess.run([model.train_op_gen, model.cost_gen],
                            feed_dict={model.lrn_rate: lrn_rate,
                                       model.z_noise: z_noise,
                                       model.label: batch_labels})
    dis_loss, p_real, p_gen = sess.run([model.cost_dis, model.p_real, model.p_fake], 
                                        feed_dict={model.lrn_rate: lrn_rate,
                                                   model.z_noise: z_noise, 
                                                   model.image: batch_images, 
                                                   model.label: batch_labels})


    ## 2. UPDATE DISCRIMINATOR
    _, dis_loss = sess.run([model.train_op_dis, model.cost_dis],
                             feed_dict={model.lrn_rate: lrn_rate,
                                        model.z_noise: z_noise,
                                        model.image: batch_images,
                                        model.label: batch_labels})
    gen_loss, p_real, p_gen = sess.run([model.cost_gen, model.p_real, model.p_fake], 
                                        feed_dict={model.lrn_rate: lrn_rate,
                                                   model.z_noise: z_noise, 
                                                   model.image: batch_images, 
                                                   model.label: batch_labels})

    if step % 5 == 0:
      print('Iteration %d, Gen Loss: %.4f, Dis Loss: %.4f (P(real)=%.2f, P(gen)=%.2f)'
            %(step, gen_loss, dis_loss, p_real.mean(), p_gen.mean()))

    ## SNAPSHOT VISUALIZATION
    if np.mod(step, 200) == 0:
      visualize_dim = 100
      z_vis_noise = np.random.uniform(-1, 1, size=(visualize_dim, z_dim))
      label_vis = onehot(np.random.randint(num_classes, size=[visualize_dim]))
      gen_samples = sess.run(model.gen_image,
                             feed_dict={model.z_noise: z_vis_noise,
                                        model.label: label_vis})
      gen_samples = (gen_samples + 1.) / 2.
      save_visualization(gen_samples, (10, 10), save_path='./vis/iter_%04d.jpg' % step)

    ## SNAPSHOT MODELS
    if step % 200 == 0:
      save_path = saver.save(sess, FLAGS.log_dir + "iter_%d" % (step))
      print('Model saved in file: %s.' % save_path)

      
  print('Optimization done.')
  print('Save the checkpoint')




def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  hps = net.HParams(batch_size=FLAGS.batch_size,
                           lrn_rate=FLAGS.base_lr,
                           weight_decay_rate=0.0001,
                           relu_leakiness=0)

  with tf.device(dev):
    train(hps)

if __name__ == '__main__':
  tf.app.run()
