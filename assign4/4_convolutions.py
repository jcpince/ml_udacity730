
# coding: utf-8

# Deep Learning
# =============
#
# Assignment 4
# ------------
#
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`,
# we trained fully connected networks to classify [notMNIST]
# (http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)
# characters.
#
# The goal of this assignment is make the neural network convolutional.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can
# import them before proceeding further.

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
from notmnist_utils import notmnist
import os
import matplotlib.pyplot as plt
# Config the matlotlib backend as plotting inline in IPython
get_ipython().magic('matplotlib qt4')

script_name = os.path.splitext(os.path.basename(__file__))[0]
pickle_file = '../data/notMNIST.pickle'

nm = notmnist(pickle_file)
nm.reshape_4d()
nm.encode_onehot()
nm.print_imports()

def relu_conv2d(x, W, b):
  return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
              padding='SAME') + b)

def max_pool(x, ksize):
  return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
              strides=[1, 2, 2, 1], padding='SAME')

def get_conv_variables(ichannels, ochannels, ksize):
  w = tf.Variable(tf.truncated_normal(
              [ksize, ksize, ichannels, ochannels], stddev=0.1))
  b = tf.Variable(tf.zeros([ochannels]))
  return w, b

def get_fully_connected_variables(isize, osize):
  w = tf.Variable(tf.truncated_normal(
              [isize, osize], stddev=0.1))
  b = tf.Variable(tf.zeros([osize]))
  return w, b


# Parameters:
batch_size = 500
initial_lr = 1e-2
lr_decay = 0.98
lr_decay_rate = 10000
c1_nfmaps = 16
c1_ksize = 5
s2_ksize = 2
c3_nfmaps = 32
c3_ksize = 5
s4_ksize = 2
fc5_units = 1024
fc6_units = 768

# Training params:
chkpt = script_name + '.chkpt'
check_frequency = 100
test_frequency = 5000

graph = tf.Graph()

with graph.as_default():
  # Input parameters.
  X = tf.placeholder(tf.float32, shape=[None, nm.image_size,
      nm.image_size, nm.num_channels])
  true_Y = tf.placeholder(tf.float32, shape=[None, nm.num_labels])
  keep_prob = tf.placeholder(tf.float32)

  # Model.
  # First layer: convolution
  c1_w, c1_b = get_conv_variables(nm.num_channels, c1_nfmaps, c1_ksize)
  c1 = relu_conv2d(X, c1_w, c1_b)

  # Second layer: subsampling
  s2 = max_pool(c1, s2_ksize)

  # Third layer: convolution
  c3_w, c3_b = get_conv_variables(c1_nfmaps, c3_nfmaps, c3_ksize)
  c3 = relu_conv2d(s2, c3_w, c3_b)

  # Fourth layer: subsampling
  s4 = max_pool(c3, s4_ksize)

  # Flatten the output
  s4_shape = s4.get_shape().as_list()
  s4_flat_cols = s4_shape[1] * s4_shape[2] * s4_shape[3]
  s4_flat = tf.reshape(s4, [-1, s4_flat_cols])

  # Fifth layer: fully connected
  fc5_w, fc5_b = get_fully_connected_variables(s4_flat_cols, fc5_units)
  fc5 = tf.nn.relu(tf.matmul(s4_flat, fc5_w) + fc5_b)

  # Sixth layer: fully connected
  fc6_w, fc6_b = get_fully_connected_variables(fc5_units, fc6_units)
  fc6 = tf.nn.relu(tf.matmul(fc5, fc6_w) + fc6_b)

  # Dropout to avoid overfitting
  fc6_drop = tf.nn.dropout(fc6, keep_prob)


  # Output layer: 10 classes
  fco_w, fco_b = get_fully_connected_variables(fc6_units, nm.num_labels)
  Y = tf.nn.relu(tf.matmul(fc6_drop, fco_w) + fco_b)

  # Training computation.
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(Y, true_Y))

  # Optimizer
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(initial_lr, global_step,
                          lr_decay_rate, lr_decay, staircase=True)
  optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss,
                          global_step=global_step)

  # Predictions of the data
  comparison = tf.equal(tf.argmax(Y,1), tf.argmax(true_Y,1))
  accuracy = tf.mul(100.0, tf.reduce_mean(tf.cast(comparison, tf.float32)))

def get_ds_accuracy(ds_name):
  global nm
  ds = nm.get_ds(ds_name)
  nb_steps = int(ds.shape[0] / batch_size)
  rest = ds.shape[0] % batch_size
  total_accy = 0.0
  for step in range(nb_steps):
    batch_data, batch_labels = nm.next_batches(ds_name, batch_size)
    total_accy += accuracy.eval(feed_dict={
          X: batch_data, true_Y: batch_labels, keep_prob: 1.0 })
  return total_accy / nb_steps

def get_ds_comparison(ds_name):
  global nm
  ds = nm.get_ds(ds_name)
  nb_steps = int(ds.shape[0] / batch_size)
  rest = ds.shape[0] % batch_size
  total_comp = np.ndarray(0, dtype=np.bool)
  for step in range(nb_steps):
    batch_data, batch_labels = nm.next_batches(ds_name, batch_size)
    total_comp = np.concatenate([total_comp, comparison.eval(feed_dict={
          X: batch_data, true_Y: batch_labels, keep_prob: 1.0 })])
  return total_comp

def train(restore_params):
  global batch_size, lr, lr_decay, chkpt
  global check_frequency, test_frequency, nm, graph
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    print('Initialized')
    if restore_params == True:
      saver.restore(session, chkpt)
    min_loss = float('inf')
    global_step = 0
    batch_loss = 0.0
    while(True):
      batch_data, batch_labels = nm.next_batches('train', batch_size)
      feed_dict = {X: batch_data, true_Y: batch_labels,
          keep_prob: 0.5}
      _, l, pred, lr = session.run([optimizer, loss, Y, learning_rate],
              feed_dict=feed_dict)
      batch_loss += l
      if (global_step != 0 and global_step % check_frequency == 0):
        batch_loss /= check_frequency
        if (batch_loss < min_loss):
          # Hit a better set of parameters, save it
          saver.save(session, chkpt)
          min_loss = batch_loss
        accy = get_ds_accuracy('valid')
        print('Minibatch loss at step %05d: %s, lr %s -- valid accuracy %s' %
            (global_step, '{:8.4f}'.format(batch_loss),
            '{:8.4f}'.format(lr), '{:5.2f}'.format(accy)))
        batch_loss = 0.0
      if (global_step != 0 and global_step % test_frequency == 0):
        accy = get_ds_accuracy('test')
        print('Test accuracy %s' % '{:5.2f}'.format(accy))
      global_step +=1
    accy = get_ds_accuracy('test')
    print('Test accuracy: %s' % '{:5.2f}'.format(accy))

def restart_training():
  train(False)

def continue_training():
  train(True)

def get_bad_predictions(ds_name):
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    print('Initialized')
    saver.restore(session, chkpt)
    comp = get_ds_comparison(ds_name)
    indices = [i for i, x in enumerate(comp) if x == False]
    return indices

def show_image_index(ds_name, index):
  global nm
  ds = nm.get_ds(ds_name)
  labels = nm.get_labels(ds_name)
  print("Letter at index %d is %s" % (index,
    chr(ord('a') + labels[index].tolist().index(1) ) ) )
  plt.figure()
  plt.imshow(ds[index].reshape(28, 28), cmap='gray')


def evaluate_cnn_on_ds(ds_name):
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    print('Initialized')
    saver.restore(session, chkpt)
    accy = get_ds_accuracy(ds_name)
    print('%s accuracy: %s' % (ds_name, '{:5.2f}'.format(accy)))
