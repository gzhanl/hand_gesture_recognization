
import tensorflow as tf
import numpy as np
import dataset as ds

'''
def generate_data():
    num = 25
    label = np.asarray(range(0, num))  # label : 0~24
    images = np.random.random((num, 5, 5, 3))
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return label, images

def get_batch_data():
    label, images = generate_data()
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False) # 默认 shuffle=True
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
    return image_batch, label_batch

image_batch, label_batch = get_batch_data()

print( image_batch )



with tf.Session() as sess:
    coord = tf.train.Coordinator()   # 开启一个协调器
    threads = tf.train.start_queue_runners(sess, coord)   # 使用start_queue_runners 启动队列填充
    i = 0
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])

            print(image_batch_v.shape, label_batch_v[i])
            i += 1
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

'''
# 样本个数
sample_num = 20
# 设置迭代次数
epoch_num = 1
# 设置一个批次中包含样本个数
#batch_size = 5
# 计算每一轮epoch中含有的batch个数
#batch_total = int(sample_num / batch_size) + 1


# 生成4个数据和标签
def generate_data(sample_num=sample_num):
    labels = np.asarray(range(sample_num))
    images = np.random.random([sample_num, 5])
    print('image data {},\n ,label {},\n,image size {},label size :{}'.format(images, labels ,images.shape, labels.shape))
    return images, labels

# 传入 全部 data 和 对应的 全部 lable 和 batch size 随机输出一批 样本
def next_batch(data,labels,batch_size):
    #images, label = generate_data()
    # 数据类型转换为tf.float32
    data = tf.cast(data, tf.float32)
    labels = tf.cast(labels, tf.int32)

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


'''
image_batch, label_batch = get_batch_data(batch_size=batch_size)

print(image_batch)
print(label_batch)
'''
train_x, train_y, train_y_onehot,test_x, test_y,test_y_onehot, test_classes=ds.load_dataset()

batch_data_x,batch_lables=next_batch(train_x,train_y_onehot,8)

print(batch_data_x)
print(batch_lables)



'''with tf.Session() as sess:
    # 先执行初始化工作
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)


    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
    # print(image_batch_v,image_batch_v.shape, label_batch_v)

    coord.request_stop()
    # print('all threads are asked to stop!')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
    # print('all threads are stopped!')
'''
