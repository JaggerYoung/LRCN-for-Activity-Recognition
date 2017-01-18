import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob
import numpy as np
import cPickle as p
import numpy as np

NUM_SAMPLES = 28
BATCH_SIZE = 15

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
    num += 1
    f = open(Filename,'r')
    total = f.readlines()
    #print len(total)
    for eachLine in range(len(total)):
        pic = []
	tmp = total[eachLine].split('\n')
	tmp_1, tmp_2 = tmp[0].split(' ',1)
	tmp_1 = '/data/zhigang.yang/UCF-101'+tmp_1
	for filename in glob.glob(tmp_1+'/*.jpg'):
	    pic.append(filename)
	len_pic = len(pic)
	l_n = len_pic/num
	data_tmp = []
	for i in range(num):
	    data_1.append(pic[i*l_n])    
	for i in range(num-1):
	    data_2.append(int(tmp_2))
    f.close()
    return (data_1, data_2)

def readImg(FileList, data_shape):
    mat = []
    tmp = 0
    ret = len(FileList)/(NUM_SAMPLES+1)
    for i in range(ret):
        for j in range(NUM_SAMPLES):
	    index = i * (NUM_SAMPLES+1) + j    
            img_1 = cv2.imread(FileList[index], 0)
            img_11 = cv2.resize(img_1, (data_shape[2], data_shape[1]))
            img_111 = np.multiply(img_11, 1/255.0)
            img_2 = cv2.imread(FileList[index+1], 0)
            img_22 = cv2.resize(img_2, (data_shape[2], data_shape[1]))
            img_222 = np.multiply(img_22, 1/255.0)
	    flow = cv2.calcOpticalFlowFarneback(img_111, img_222, 0.5, 3, 15, 3, 5, 1.2, 0)
	    flow = np.array(flow)
            flow_1 = flow.transpose((2,1,0))
	    flow_1 = flow_1.tolist()
	    mat.append(flow_1)
    return mat

class VGGIter(mx.io.DataIter):
    def __init__(self, fname, num, batch_size, data_shape):
        self.batch_size = batch_size
	self.fname = fname
	self.data_shape = data_shape
	self.num = num*NUM_SAMPLES/batch_size
	(self.data_1, self.data_2) = readData(fname, NUM_SAMPLES)
	self.img = readImg(self.data_1, self.data_shape)
    
        self.provide_data = [('data', (batch_size,) + data_shape)]
	self.provide_label = [('label', (batch_size,))]

    def __iter__(self):
        for k in range(self.num):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * self.batch_size + i
		data.append(self.img[idx])
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
    data_shape = (2, 224, 224)
    
    train_file = '/home/users/zhigang.yang/mxnet/example/LRCN-for-Activity-Recognition/data/train.list'
    test_file = '/home/users/zhigang.yang/mxnet/example/LRCN-for-Activity-Recognition/data/test.list'

    
    data_train = VGGIter(train_file, train_num, batch_size, data_shape)
    data_val = VGGIter(test_file, test_num, batch_size, data_shape)

    print data_train.provide_data, data_train.provide_label

    devs = [mx.context.gpu(1)]
    model = mx.model.FeedForward.load("./vgg_model/vgg16", epoch=00, ctx=devs, num_batch_size=BATCH_SIZE)

    internals = model.symbol.get_internals()
    print internals.list_outputs()
    fea_symbol = internals['relu7_output']
    feature_exactor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, num_batch_size=1,
                                           arg_params=model.arg_params, aux_params=model.aux_params,
    					   allow_extra_params=True)
    vgg_train_result = feature_exactor.predict(data_train)
    vgg_test_result = feature_exactor.predict(data_val)
    
    print mx.nd.array(vgg_train_result).shape
    print mx.nd.array(vgg_test_result).shape
    #return (vgg_train_result, vgg_test_result)
    train_data_file = 'train_data.data'
    f_1 = file(train_data_file, 'w')
    p.dump(vgg_train_result, f_1)
    f_1.close()

    test_data_file = 'test_data.data'
    f_2 = file(test_data_file, 'w')
    p.dump(vgg_test_result, f_2)
    f_2.close()

#def get_label():
    
    (tmp_1, train_label) = readData(train_file, NUM_SAMPLES)
    (tmp_2, test_label) = readData(test_file, NUM_SAMPLES)

    print mx.nd.array(train_label).shape
    print mx.nd.array(test_label).shape
#   return (train_label, test_label)
    train_label_file = 'train_label.data'
    f_3 = file(train_label_file, 'w')
    p.dump(train_label, f_3)
    f_3.close()

    test_label_file = 'test_label.data'
    f_4 = file(test_label_file, 'w')
    p.dump(test_label, f_4)
    f_4.close()

