import numpy as np
from datetime import datetime
import os
import math
import random as rand
import tensorflow as tf
import matplotlib.pyplot as plot
from glob import glob

def Spatial_Transfrom (inMap, theta, out_dims=None, **args):
    with tf.name_scope('Spatial_Transfrom'): 
        print("inmap{},  shape{}, shapeList{}".format(inMap, tf.shape(inMap), inMap.get_shape().as_list()))
        shape = inMap.get_shape()
        B = tf.shape(inMap)[0]
        H = shape.as_list()[1]
        W = shape.as_list()[2]
        C = shape.as_list()[3]
        
        print('b:{}'.format(B))
        #print('w:{}, shape:{}'.format(w, w.get_shape().as_list()))
        
        #construct theta matrix
        theta = tf.reshape(theta, [B, 2, 3], name='theta')
        
        if out_dims:
            out_H = out_dims[0]
            out_W = out_dims[1]
            batch_grids = affine_grid_gen(out_H, out_W, theta)
        else:
            batch_grids = affine_grid_gen(H, W, theta)

        batch_grids = tf.identity(batch_grids, name='batch_grids')
        
        print('batch_grids')
        print(batch_grids.get_shape().as_list())
        xs = batch_grids[:, 0, :, :]
        ys = batch_grids[:, 1, :, :]
        
        out_samp = bilinear_sampler(inMap, xs, ys)
        out_samp = tf.identity(out_samp, name='bilinear_sampler')
        print('Spatial_Transfrom')
        print(inMap.get_shape().as_list())
        print(out_samp.get_shape().as_list())
        print('\n')
        return out_samp

def affine_grid_gen(height, width, theta):
    print('affine_grid_gen, height:{}, width:{}, theta:{}'.format(height, width, theta.get_shape().as_list()))
    
    num_batch = tf.shape(theta)[0]
    
    #nornalized grid elements
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)
    print('affine_grid_gen, x:{}, y:{}, x_t:{}, y_t:{}'.format(x.get_shape().as_list(), y.get_shape().as_list(), 
                                                               x_t.get_shape().as_list(), y_t.get_shape().as_list()))
    
    #flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    print('affine_grid_gen, x_t_flat:{}, y_t_flat:{}'.format(x_t_flat.get_shape().as_list(), y_t_flat.get_shape().as_list()))
    
    #generate grid
    ones = tf.ones_like(x_t_flat)
    print('ones')
    print(ones.get_shape().as_list())
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))
    
    #cast to float32
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')
    
    batch_grids = tf.matmul(theta, sampling_grid)
    
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])
    
    return batch_grids

def bilinear_sampler(img, x, y):
    print("bilinear_sampler x:{}, y:{}".format(str(x), str(y)))
    
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]
    
    max_y = tf.cast(H -1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))
    
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)
    
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)