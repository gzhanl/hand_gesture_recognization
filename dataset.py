
import numpy as np
import h5py
import tensorflow as tf


'''
h5py文件是存放两类对象的容器，数据集(dataset)和组(group)，dataset类似数组类的数据集合，和numpy的数组差不多。
group是像文件夹一样的容器，它好比python中的字典，有键(key)和值(value)。
group中可以存放dataset或者其他的group。
”key”就是组成员的名称，”value”就是组成员对象本身(组或者数据集)，下面来看下如何创建组和数据集。
'''

def load_dataset():

    train_dataset = h5py.File('datasets/train_signs.h5', 'r')

    # 查看 train_dataset 里面有什么
    for key in train_dataset.keys():
        print("train_dataset key: ",key)    # list_classes    train_set_x    train_set_y
        #print("train_dataset key name : ",train_dataset[key].name)
        #print("The shape of %s : \n" % ( key ),train_dataset[key].shape)
        #print("The value of %s : \n" % ( key ),train_dataset[key].value)

    '''
    train_dataset key:  list_classes
    train_dataset key name :  /list_classes
    The shape of list_classes : 
     (6,)
    The value of list_classes : 
     [0 1 2 3 4 5]
     
    train_dataset key:  train_set_x
    train_dataset key name :  /train_set_x
    The shape of train_set_x : 
     (1080, 64, 64, 3)
    The value of train_set_x : 
     [[[[227 220 214]
       [227 221 215]
       [227 222 215]
       ..., 
       [232 230 224]
       [231 229 222]
       [230 229 221]]
    
      .....
    
      [[214 205 198]
       [216 206 199]
       [217 207 199]
       ..., 
       [201 198 197]
       [202 199 198]
       [202 199 197]]]]
       
    train_dataset key:  train_set_y
    train_dataset key name :  /train_set_y
    The shape of train_set_y : 
     (1080,)
    The value of train_set_y : 
     [5 0 2 ..., 2 4 5]
    '''

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])      #  " train_set_x" 是 train_dataset 的 key name ，不是随便定义的
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])      # [ : ] 表示全部   0 ~ 1079 即 ：1080个
    # print(train_set_x_orig)
    '''
    [[[[227 220 214]
       [227 221 215]
       [227 222 215]
       ..., 
       [232 230 224]
       [231 229 222]
       [230 229 221]]
    
      .....
    
      [[214 205 198]
       [216 206 199]
       [217 207 199]
       ..., 
       [201 198 197]
       [202 199 198]
       [202 199 197]]]]
    '''

    # print(train_set_y_orig)
    '''
    [5 0 2 ..., 2 4 5]
    '''


    # 将 train_set_y_orig 进行 one hot 编码
    train_y_onehot_process=tf.one_hot(train_set_y_orig,6)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_y_onehot = sess.run(train_y_onehot_process)

    #print(train_y_onehot)
    #print(train_y_onehot.shape)
    '''
    [[ 0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.]
     [ 0.  0.  1.  0.  0.  0.]
     ..., 
     [ 0.  0.  1.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.]
     [ 0.  0.  0.  0.  0.  1.]]
   
    (1080, 6)
    '''


   ##  test_dataset 内容与 train_dataset 结构一样
    test_dataset = h5py.File('datasets/test_signs.h5', 'r')

    # 查看 test_dataset 里面有什么
    for key in test_dataset.keys():
        print("test_dataset key: ", key)
        #print("test_dataset key name : ", test_dataset[key].name)
        #print("The shape of %s : \n" % ( key ),test_dataset[key].shape)
        #print("The value of %s : \n" % ( key ),test_dataset[key].value)

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    test_y_onehot_process = tf.one_hot(test_set_y_orig, 6)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_y_onehot = sess.run(test_y_onehot_process)

    test_classes = np.array(test_dataset["list_classes"][:])    # [0 1 2 3 4 5]


    #----------将test 和 train 的 y shape 从（120，）转为 （1，120）--------------------

    print("before reshape: ", test_set_y_orig.shape, test_set_y_orig)

    train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])  # 从 （120，）转为 （1，120）

    print("after  reshape: ", test_set_y_orig.shape,test_set_y_orig)

    return train_set_x_orig, train_set_y_orig, train_y_onehot,test_set_x_orig, test_set_y_orig,test_y_onehot, test_classes




# 传入 全部 data 和 对应的 全部 lable 和 batch size 随机输出一批 样本
def next_batch(data,labels,batch_size):

    # 数据类型转换为tf.float32
    #data = tf.cast(data, tf.float32)
    #labels = tf.cast(labels, tf.int32)

    # 从tensor列表中按顺序或随机抽取一个tensor准备放入文件名称队列    # num_epochs=1 shuffle
    input_queue = tf.train.slice_input_producer([data, labels], num_epochs=1, shuffle=True)

    # 从文件名称队列中读取文件准备放入文件队列
    data_batch, labels_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=64,
                                              allow_smaller_final_batch=True)

    with tf.Session() as sess:
        # 先执行初始化工作
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 开启一个协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)

        data_batch, labels_batch = sess.run([data_batch, labels_batch])
        # print(image_batch_v,image_batch_v.shape, label_batch_v)

        coord.request_stop()
        # print('all threads are asked to stop!')
        coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
        # print('all threads are stopped!')

    return data_batch, labels_batch


if __name__ == '__main__':

    ds=load_dataset()

