

import numpy as np
from scipy.misc import imsave
from skimage import transform
import os
import dataset as ds

def data2jpg():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, test_classes = ds.load_dataset()

    # print("X_train_orig shape: " + str(X_train_orig.shape))
    # print("Y_train_orig shape: " + str(Y_train_orig.shape))
    # print("X_test_orig shape: " + str(X_test_orig.shape))
    # print("Y_test_orig shape: " + str(Y_test_orig.shape))

    # print(classes[1])

    m = len(X_train_orig)
    # print(X_train_orig[1].shape)

    Y_train_t = Y_train_orig.T

    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.imshow(X_train_orig[i])
    #     plt.title(Y_train_t[i])
    #     plt.axis('off')
    #
    # plt.show()


    for i in range(m):
            if not os.path.exists('images/train/'):
                os.makedirs('images/train/')
            name = 'images/train/' + str(i) + '-[' + str(np.squeeze(Y_train_t[i])) + '].jpg'
            # name = 'images/train/' + str(i) + '.jpg'
            imsave(name, transform.rescale(X_train_orig[i].reshape(64, 64, 3), 10, mode='constant'))  # (640, 640, 3)


    test_m = len(X_test_orig)

    for j in range(test_m):
        if not os.path.exists('images/test/'):
            os.makedirs('images/test/')
        name = 'images/test/' + str(j) + '.jpg'
        # name = 'images/train/' + str(i) + '.jpg'
        imsave(name, transform.rescale(X_test_orig[j].reshape(64, 64, 3), 10, mode='constant'))  # (640, 640, 3)

