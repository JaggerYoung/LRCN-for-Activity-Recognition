import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob
import numpy as np
import cPickle as p

NUM_SAMPLES = 28
BATCH_SIZE = 28

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

	self.pad = 0
	self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n,x in zip(self.label_names, self.label)]

def readData(Filename, num):
    data_1 = []
    data_2 = []
    f = open(Filename,'r')
    total = f.readlines()

    for eachLine in range(len(total)):
        pic = []
        tmp = total[eachLine].split('\n')
	tmp_1, tmp_2 = tmp[0].split(' ',1)
	tmp_1 = '/home/yzg/UCF-101'+tmp_1
	for filename in glob.glob(tmp_1+'/*.jpg'):
	    pic.append(filename)
	len_pic = len(pic)
	l_n = len_pic/num
	for i in range(num):
	    data_1.append(pic[i*l_n])    
	    data_2.append(int(tmp_2))
    f.close()
    return (data_1, data_2)

def readImg(Filename, data_shape):
    mat = []

    img = cv2.imread(Filename, cv2.IMREAD_COLOR)
    r,g,b = cv2.split(img)
    r = cv2.resize(r, (data_shape[2], data_shape[1]))
    g = cv2.resize(g, (data_shape[2], data_shape[1]))
    b = cv2.resize(b, (data_shape[2], data_shape[1]))
    r = np.multiply(r, 1/255.0)
    g = np.multiply(g, 1/255.0)
    b = np.multiply(b, 1/255.0)

    mat.append(r)
    mat.append(g)
    mat.append(b)

    return mat

class VGGIter(mx.io.DataIter):
    def __init__(self, fname, num, batch_size, data_shape):
        self.batch_size = batch_size
	self.fname = fname
	self.data_shape = data_shape
	self.num = num*NUM_SAMPLES/batch_size
	(self.data_1, self.data_2) = readData(fname, NUM_SAMPLES)
    
        self.provide_data = [('data', (batch_size,) + data_shape)]
	self.provide_label = [('label', (batch_size,))]

    def __iter__(self):
        for k in range(self.num):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * self.batch_size + i
		img = readImg(self.data_1[idx], self.data_shape)
		data.append(img)
	        label.append(self.data_2[k])
	
	    data_all = [mx.nd.array(data)]
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch
    
    def reset(self):
        pass

if __name__ == '__main__':
#def vgg_predict():    
    train_num = 9537
    test_num = 3783

    batch_size = BATCH_SIZE
    data_shape = (3, 224, 224)
    
    train_file = '/home/yzg/mxnet/example/LRCN_UCF101/data/train.list'
    test_file = '/home/yzg/mxnet/example/LRCN_UCF101/data/test.list'

    data_train = VGGIter(train_file, train_num, batch_size, data_shape)
    data_val = VGGIter(test_file, test_num, batch_size, data_shape)

    print data_train.provide_data, data_train.provide_label

    devs = [mx.context.gpu(0)]
    model = mx.model.FeedForward.load("./vgg_model/vgg16", epoch=00, ctx=devs, num_batch_size=BATCH_SIZE)

    internals = model.symbol.get_internals()
    print internals.list_outputs()
    fea_symbol = internals['relu7_output']
    feature_exactor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, num_batch_size=1,
                                           arg_params=model.arg_params, aux_params=model.aux_params,
					   allow_extra_params=True)
    vgg_train_result = feature_exactor.predict(data_train)
    vgg_test_result = feature_exactor.predict(data_val)
    
    #print mx.nd.array(vgg_train_result).shape
    #return (vgg_train_result, vgg_test_result)
    train_data_file = 'train_data.data'
    f_1 = file(train_data_file, 'w')
    p.dump(vgg_train_result, f_1)
    f_1.close()

    test_data_file = 'test_data.data'
    f_2 = file(test_data_file, 'w')
    p.dump(vgg_test_result, f_2)
    f2.close()

#def get_label():
    train_file = '/home/yzg/mxnet/example/LRCN_UCF101/data/train.list'
    test_file = '/home/yzg/mxnet/example/LRCN_UCF101/data/test.list'
    
    (tmp_1, train_label) = readData(train_file, NUM_SAMPLES)
    (tmp_2, test_label) = readData(test_file, NUM_SAMPLES)

#    return (train_label, test_label)
    train_label_file = 'train_label.data'
    f_3 = file(train_label_file, 'w')
    p.dump(train_label, f_3)
    f_3.close()

    test_label_file = 'test_label.data'
    f_2 = file(test_label_file, 'w')
    p.dump(test_label, f_2)
    f2.close()

