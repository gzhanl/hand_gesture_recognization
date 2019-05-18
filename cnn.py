import math
import types
import numpy as np
import h5py
import cv2
from skimage import transform

import matplotlib.pyplot as plt
import tensorflow as tf

import dataset as ds

'''
def reading():
    image = cv2.imread('images/train/16-[1].jpg', cv2.IMREAD_UNCHANGED)

    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('input_image', transform.rescale(image, 0.5, mode='constant'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''


def recogization_result():


    print('recogize Finish ')



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def get_Batch(data, label, batch_size):
    print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32,
                                      allow_smaller_final_batch=False)
    return x_batch, y_batch

if __name__ == '__main__':
    # reading()

    train_x, train_y, train_y_onehot,test_x, test_y,test_y_onehot, test_classes=ds.load_dataset()
    #print(train_x.shape) # (1080, 64, 64, 3)
    #print(train_y.shape) # (1, 1080)
    #print(train_y_onehot.shape) # (1080, 6)
    #data2jpg()

    
    # x为训练图像的占位符、y_为训练图像标签的占位符
    #x = tf.placeholder(tf.float32, [None, 4096])  # 4096 = 64 * 64       x: (1080, 64, 64, 3)
    #y_ = tf.placeholder(tf.float32, [None, 6])

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    #x_image = tf.reshape(x, [-1, 64, 64, 3])
    #print("image shape: ", x_image)

    x = tf.placeholder(tf.float32, shape=[None, 64,64,3])
    y_ = tf.placeholder(tf.float32, shape=[None, 6])
    x_image = x

    # 第一层卷积层
    W_conv1 = weight_variable([5, 5, 3, 68])  # 68=64+5-1
    b_conv1 = bias_variable([68])             # 同 68
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([5, 5, 68, 136])   # 136 = 68 * 2
    b_conv2 = bias_variable([136])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层，输出为1024维的向量        # 改为768或其他也可以运行  如：W_fc1 = weight_variable([7 * 7 * 64, 768]) 下面相关数值也要改
    W_fc1 = weight_variable([16 * 16 * 136, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 136])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 把1024维的向量转换成10维，对应10个类别
    W_fc2 = weight_variable([1024, 6])
    b_fc2 = bias_variable([6])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 同样定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    # 训练20000步
    for i in range(2000):


        batch_x,batch_lables=ds.next_batch(train_x, train_y_onehot,50)



        # 每100步报告一次在验证集上的准确度
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_lables, keep_prob: 1.0})  #
            #print(batch_x.shape)  #  (1080, 64, 64, 3)
            #print(batch_lables.shape)  #   (1080, 6)

            print("step %d, training accuracy %f" % (i, train_accuracy))
            #print("step %d" % (i))
        train_step.run(feed_dict={x: batch_x, y_:batch_lables , keep_prob: 0.5})

    # 训练结束后报告在测试集上的准确度
    # print("test accuracy %g" % accuracy.eval(feed_dict={
    #     x: train_x.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    print('finish!')
