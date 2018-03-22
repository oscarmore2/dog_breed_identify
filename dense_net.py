import numpy as np
import tensorflow as tf
import os


from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.train.base import *

LayerGrowth =12

def Batch_Normalization(scope, x):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return batch_norm(inputs=x, reuse=None)

def conv(name, layer, channel, stride):
    return Conv2D(name, layer, channel, 3, stride=stride,
                  nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))

def add_transition(name, input_):
    shape = input_.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        input_ = Batch_Normalization('bn1', input_)
        input_ = tf.nn.relu(input_)
        input_ = Conv2D('conv1', input_, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        input_ = AvgPooling('pool', input_, 2)
    return input_


def add_layer(name, layer):
    shape = layer.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        c = Batch_Normalization('bn1', layer)
        c = tf.nn.relu(c)
        c = conv('conv1', c, LayerGrowth, 1)
        layer = tf.concat([c, layer], 3)
    return layer

def Fully_connected(x, num_classes, layer_name='') :
    with tf.name_scope('fully_connected'+layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=num_classes)

def Dense_net(name, input, num_classes, n):

    layer = conv('conv0_'+name, input, 16, 1)
    with tf.variable_scope('block1_'+name) as scope:
        for i in range(n):
            layer = add_layer('dense_layer.{}'.format(i), input);
        layer = add_transition('transition1', layer);

    with tf.variable_scope('block2_'+name) as scope:
        for i in range(n):
            layer = add_layer('dense_layer.{}'.format(i), input);
        layer = add_transition('transition2', layer)

    with tf.variable_scope('block3_'+name) as scope:
        for i in range(n):
            layer = add_layer('dense_layer.{}'.format(i), input);

    layer = Batch_Normalization('bnlast_'+name, layer)
    layer = tf.nn.relu(layer)
    layer = GlobalAvgPooling('gap_'+name, layer)
    logits = Fully_connected(layer, num_classes, name)

    return logits