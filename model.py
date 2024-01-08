from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *
import logging

# ログの設定
logging.basicConfig(
    filename='app.log',  # ログを保存するファイル名
    level=logging.WARN,  # ログのレベル (INFO, WARNING, ERROR, DEBUGなど)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def FG(input_im):
  with tf.compat.v1.variable_scope('FG'):
    input_rs = tf.image.resize(input_im, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    p_conv1 = tf.compat.v1.layers.conv2d(input_rs, 64, 3, 2, padding='same', activation=tf.nn.relu) # 48
    p_conv2 = tf.compat.v1.layers.conv2d(p_conv1,  64, 3, 2, padding='same', activation=tf.nn.relu) # 24
    p_conv3 = tf.compat.v1.layers.conv2d(p_conv2,  64, 3, 2, padding='same', activation=tf.nn.relu) # 12
    p_conv4 = tf.compat.v1.layers.conv2d(p_conv3,  64, 3, 2, padding='same', activation=tf.nn.relu) # 6
    p_conv5 = tf.compat.v1.layers.conv2d(p_conv4,  64, 3, 2, padding='same', activation=tf.nn.relu) # 3
    p_conv6 = tf.compat.v1.layers.conv2d(p_conv5,  64, 3, 2, padding='same', activation=tf.nn.relu) # 1

    p_deconv1 = tf.image.resize(p_conv6, (3, 3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv1 = tf.compat.v1.layers.conv2d(p_deconv1, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv1 = p_deconv1 + p_conv5
    p_deconv2 = tf.image.resize(p_deconv1, (6, 6), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv2 = tf.compat.v1.layers.conv2d(p_deconv2, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv2 = p_deconv2 + p_conv4
    p_deconv3 = tf.image.resize(p_deconv2, (12, 12), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv3 = tf.compat.v1.layers.conv2d(p_deconv3, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv3 = p_deconv3 + p_conv3
    p_deconv4 = tf.image.resize(p_deconv3, (24, 24), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv4 = tf.compat.v1.layers.conv2d(p_deconv4, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv4 = p_deconv4 + p_conv2
    p_deconv5 = tf.image.resize(p_deconv4, (48, 48), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv5 = tf.compat.v1.layers.conv2d(p_deconv5, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv5 = p_deconv5 + p_conv1
    p_deconv6 = tf.image.resize(p_deconv5, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv6 = tf.compat.v1.layers.conv2d(p_deconv6, 64, 3, 1, padding='same', activation=tf.nn.relu)

    p_output = tf.image.resize(p_deconv6, (tf.shape(input=input_im)[1], tf.shape(input=input_im)[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    a_input = tf.concat([p_output, input_im], axis=3)
    a_conv1 = tf.compat.v1.layers.conv2d(a_input, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv2 = tf.compat.v1.layers.conv2d(a_conv1, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv3 = tf.compat.v1.layers.conv2d(a_conv2, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv4 = tf.compat.v1.layers.conv2d(a_conv3, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv5 = tf.compat.v1.layers.conv2d(a_conv4, 3,   3, 1, padding='same', activation=tf.nn.relu)
    return a_conv5



class lowlight_enhance(object):
    def __init__(self, sess):
        
        self.sess = sess
        self.base_lr = 0.001
        #self.g_window = self.gaussian_window(self.input_shape[0],self.input_shape[2],0.5)
        self.input_low = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_high')
        #self.norm_const = self.input_low[2]*self.batch_size
        self.output = FG(self.input_low)
        self.loss = tf.reduce_mean(input_tensor=tf.abs((self.output - self.input_high) * [[[[0.11448, 0.58661, 0.29891]]]]))

        self.global_step = tf.Variable(0, trainable = False)
        self.lr = tf.compat.v1.train.exponential_decay(self.base_lr, self.global_step, 100, 0.96)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        print("[*] Initialize model successfully...")
        


    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir):
        tf.compat.v1.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status, _ = self.load(self.saver, './model/')
        if load_model_status:
            print("[*] Load weights successfully...")
            pass
        
        print("[*] Testing...")
        total_run_time = 0.0
        for idx in range(len(test_low_data)):
            #print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            if not os.path.exists(os.path.join(save_dir, name +"."+ suffix)):
                print("作成中")
                input_low_test = np.expand_dims(test_low_data[idx], axis=0)
                start_time = time.time()
                result = self.sess.run(self.output, feed_dict = {self.input_low: input_low_test})
                total_run_time += time.time() - start_time
                save_images(os.path.join(save_dir, name +"."+ suffix), result)

