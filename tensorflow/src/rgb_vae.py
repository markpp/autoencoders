# ConvVAE model

import numpy as np
import json
import tensorflow as tf
import os

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.compat.v1.reset_default_graph()

class ConvVAE(object):
  def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance
    self.reuse = reuse
    with tf.compat.v1.variable_scope('conv_vae', reuse=self.reuse):
      if not gpu_mode:
        with tf.compat.v1.device('/cpu:0'):
          tf.compat.v1.logging.info('Model using cpu.')
          self._build_graph()
      else:
        tf.compat.v1.logging.info('Model using gpu.')
        self._build_graph()
    self._init_session()
  def _build_graph(self):
    self.g = tf.compat.v1.Graph()
    with self.g.as_default():

      self.x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 64, 64, 3])
      print("input: {}".format(self.x))
      # Encoder
      h = tf.compat.v1.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.compat.v1.nn.relu, name="enc_conv1")
      print(h.shape)
      h = tf.compat.v1.layers.conv2d(h, 64, 4, strides=2, activation=tf.compat.v1.nn.relu, name="enc_conv2")
      print(h.shape)
      h = tf.compat.v1.layers.conv2d(h, 128, 4, strides=2, activation=tf.compat.v1.nn.relu, name="enc_conv3")
      print(h.shape)
      h = tf.compat.v1.layers.conv2d(h, 256, 4, strides=2, activation=tf.compat.v1.nn.relu, name="enc_conv4")
      print(h.shape)
      h = tf.compat.v1.reshape(h, [-1, 2*2*256])
      print(h.shape)
      # VAE
      self.mu = tf.compat.v1.layers.dense(h, self.z_size, name="enc_fc_mu")
      self.logvar = tf.compat.v1.layers.dense(h, self.z_size, name="enc_fc_log_var")
      self.sigma = tf.compat.v1.exp(self.logvar / 2.0)
      self.epsilon = tf.compat.v1.random_normal([self.batch_size, self.z_size])
      self.z = self.mu + self.sigma * self.epsilon
      print("z: {}".format(self.z.shape))

      # Decoder
      h = tf.compat.v1.layers.dense(self.z, 4*256, name="dec_fc")
      print(h.shape)
      h = tf.compat.v1.reshape(h, [-1, 1, 1, 4*256])
      print(h.shape)
      h = tf.compat.v1.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.compat.v1.nn.relu, name="dec_deconv1")
      print(h.shape)
      h = tf.compat.v1.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.compat.v1.nn.relu, name="dec_deconv2")
      print(h.shape)
      h = tf.compat.v1.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.compat.v1.nn.relu, name="dec_deconv3")
      print(h.shape)
      self.y = tf.compat.v1.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.compat.v1.nn.sigmoid, name="dec_deconv4")
      print("output: {}".format(self.y))

      # train ops
      if self.is_training:
        self.global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)

        eps = 1e-6 # avoid taking log of zero

        # reconstruction loss
        self.r_loss = tf.compat.v1.reduce_sum(
          tf.compat.v1.square(self.x - self.y),
          reduction_indices = [1,2,3]
        )
        self.r_loss = tf.compat.v1.reduce_mean(self.r_loss)

        # augmented kl loss per dim
        self.kl_loss = - 0.5 * tf.compat.v1.reduce_sum(
          (1 + self.logvar - tf.compat.v1.square(self.mu) - tf.compat.v1.exp(self.logvar)),
          reduction_indices = 1
        )
        self.kl_loss = tf.compat.v1.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
        self.kl_loss = tf.compat.v1.reduce_mean(self.kl_loss)

        self.loss = self.r_loss + self.kl_loss

        # training
        self.lr = tf.compat.v1.Variable(self.learning_rate, trainable=False)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

        self.train_op = self.optimizer.apply_gradients(
          grads, global_step=self.global_step, name='train_step')

      # initialize vars
      self.init = tf.compat.v1.global_variables_initializer()

  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True

    self.sess = tf.compat.v1.Session(graph=self.g,config=config)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def encode(self, x):
    return self.sess.run(self.z, feed_dict={self.x: x})
  def encode_mu_logvar(self, x):
    (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
    return mu, logvar
  def decode(self, z):
    return self.sess.run(self.y, feed_dict={self.z: z})
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.compat.v1.trainable_variables()
      for var in t_vars:
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
    return rparam
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.compat.v1.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = self.sess.run(var).shape
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op = var.assign(p.astype(np.float)/10000.)
        self.sess.run(assign_op)
        idx += 1
  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='vae.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vae')
    tf.compat.v1.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.compat.v1.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
