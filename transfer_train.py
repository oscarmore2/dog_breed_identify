#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# create by Oscar
# Transfer learning of inception-v3
# 

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import pickle
import pandas
from sklearn import preprocessing


BOTTLENECK_TENSOR_SIZE = 2048


BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'


JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


MODEL_DIR = 'model/'


MODEL_FILE = 'tensorflow_inception_graph.pb'


CACHE_DIR = 'tmp/bottleneck/'


INPUT_DATA = './Data/Standford/Images/'

TEST_DATA = './Data/postTest/'


VALIDATION_PERCENTAGE = 10

TEST_PERCENTAGE = 10

LEARNING_RATE = 0.00005
STEPS = 20000
BATCH = 300

RMSPROP_DECAY = 0.9                
RMSPROP_MOMENTUM = 0.9             
RMSPROP_EPSILON = 1.0              
initial_learning_rate = 0.01
num_epochs_per_decay = 30.0
learning_rate_decay_factor = 0.16
MOVING_AVERAGE_DECAY = 0.9999



def create_image_lists(folder, testing_percentage, validation_percentage):

    result = {}

    sub_dirs = [x[0] for x in os.walk(folder)]
    
    

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        
        extensions = ['jpg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        
        label_name = dir_name.lower()
        
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        print('label_name {} has {} trainning, {} testing samples'.format(label_name, len(training_images), len(testing_images)))

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
            }

    return result



def get_image_path(image_lists, image_dir, label_name, index, category):
   
    label_lists = image_lists[label_name]
    
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path



def get_bottlenect_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt';



def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values




def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottlenect_path(image_lists, label_name, index, category)
    
    if not os.path.exists(bottleneck_path):
        
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        
        image_data = gfile.FastGFile(image_path, 'rb').read()
        
        
        
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    
    return bottleneck_values



def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        
        
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category,
                                              jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths



def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                                                  jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype = np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def Fully_connected(x, num_classes, layer_name='') :
    with tf.name_scope('fully_connected'+layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=num_classes)

def predict_test_data(sess, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    files =  [x[2] for x in os.walk(TEST_DATA)]
    img_name = []
    
    for img_file in files[0]:
        image_data = gfile.FastGFile(os.path.join(TEST_DATA, img_file), 'rb').read()
        
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck_values)
        img_name.append(os.path.splitext(img_file)[0])
    return bottlenecks, img_name

def main(_):
    
    image_lists = create_image_lists(INPUT_DATA, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    print('number of n class is {}'.format(n_classes))
    
    
    
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    
    
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    
    
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        
        final_tensor = tf.nn.softmax(logits)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    
    global_step = tf.train.get_or_create_global_step()
    
    
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    STEPS,
                                    learning_rate_decay_factor,
                                    staircase=True)
    opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY, momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)
    grads = opt.compute_gradients(cross_entropy_mean)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_to_average = (tf.trainable_variables() +  tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    train_step =  tf.group(apply_gradient_op, variables_averages_op)
    
    
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(STEPS):
            
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)

            
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            
            if i%100 == 0 or i+1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input:validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                      % (i, BATCH, validation_accuracy*100))
        
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                       jpeg_data_tensor, bottleneck_tensor)
        print(len(test_bottlenecks))
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                                 ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

        prediction_bottleneck, img_names = predict_test_data(sess, jpeg_data_tensor, bottleneck_tensor)
        print(len(prediction_bottleneck))
        lab_prediction = sess.run(final_tensor, feed_dict={bottleneck_input:prediction_bottleneck})
        min_max_scaler = preprocessing.MinMaxScaler()
        

        print(len(lab_prediction))
        print('Final Predictions is')
        print(lab_prediction)
        tab_Name = [] 
        for item in list(image_lists.keys()):
            name = item
            name = (item.split('-')[1:])
            tag = name[0]
            if len(name) > 1:
                for n in name[1:]:
                    tag = tag+'-'+n
            tab_Name.append(tag)
        df = pandas.DataFrame(lab_prediction, index=img_names, columns=tab_Name)
        df.to_csv('./whole_v3.csv')


if __name__ == '__main__':
    tf.app.run()