import numpy as np
import tensorflow as tf
import dataset as ds

def next_batch(data,lable,batch_size):
    #datasets = ds.load_dataset()[0]
    input_queue = tf.train.slice_input_producer([data,lable], shuffle=False, num_epochs=1)
    x_batch,y_batch = tf.train.batch(input_queue, batch_size, num_threads=1,
                                 capacity=20, allow_smaller_final_batch=False)
    return x_batch,y_batch




if __name__ == "__main__":
    data_x=ds.load_dataset()[0]
    label_onehot=ds.load_dataset()[2]
    x_batch ,y_batch = next_batch(data_x,label_onehot,5)


    sess = tf.Session()
    sess.run(tf.initialize_local_variables())
    #sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    while not coord.should_stop():
        data,lable = sess.run([x_batch ,y_batch])
        print(data,lable)

    coord.request_stop()
    coord.join(threads)
    sess.close()
