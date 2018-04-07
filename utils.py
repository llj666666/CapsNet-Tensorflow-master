import os
import scipy
import numpy as np
import tensorflow as tf

import scipy.misc as misc



def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        # fd = open(os.path.join(path, 'train-images-idx3-ubyte.gz'))#打开数据文件,这里后缀需要注意修改
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))#打开数据文件,这里后缀需要注意修改
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_myimg(batch_size, is_training=True):
    if is_training:
        #目录
        apath="data/myimg/"
        imgall=[]
        y=[]
        for i in range(5):
            dirname="附件%d/"%(i+1)
            list_img=os.listdir(apath+dirname)
            #排序
            if i!=4:
                list_img.sort(key=lambda x: int(x[4:-4]))
            else:
                list_img.sort()

            #读取图片
            for j in range(len(list_img)):
                try:
                    fd=apath+dirname+list_img[j]
                    # loaded = np.fromfile(file=fd, dtype=np.uint8)
                    img_in=misc.imread(fd)

                    # imgall.append(loaded)
                    imgall.append(img_in)
                    y.append([i,j%3])#标签
                    # i
                    # 0=碱度0.6
                    # 1=碱度0.8
                    # 2=碱度1.0，
                    # 3=碱度1.2
                    #
                    # j%3
                    # 0=中心部位
                    # 1=1/4部位
                    # 2=边缘部位



                except:
                    pass



        # trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trainX=imgall[:72]#前3个文件夹训练
        trainY=y[:72]

        trX = trainX / 255.
        trY = trainY

        valX = imgall[72:72+24 ] / 255.#附件4做为验证集
        valY = y[72:72+24]

        num_tr_batch = 72 // batch_size
        num_val_batch = 24 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    # else:
        # fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        # teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        #
        # fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        # teY = loaded[8:].reshape((10000)).astype(np.int32)
        #
        # num_te_batch = 10000 // batch_size
        # return teX / 255., teY, num_te_batch





def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'myimg':
        return load_myimg(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset=="myimg":
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
