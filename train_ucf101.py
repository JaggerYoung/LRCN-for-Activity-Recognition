import os,sys
import random
import find_mxnet
import mxnet as mx
import string
import math

from lstm import lstm_unroll
from cnn_predict import vgg_predict
from cnn_predict import get_label

BATCH_SIZE = 15

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.label_names, self.label)]

def Accuracy(label, pred):
    SEQ_LEN = 28
    hit = 0.
    total = 0.
    label = label.T.reshape(-1,1)
    for i in range(BATCH_SIZE*SEQ_LEN):
        maxIdx = np.argmax(pred[i])
	if maxIdx == int(label[i]):
	    hit += 1.0
	total += 1.0
    return hit/total

class LRCNIter(mx.io.DataIter):
    def __init__(self, dataset, labelset, num, batch_size, seq_len, init_states):
        
	self.batch_size = batch_size
	self.count = num/batch_size
	self.seq_len = seq_len
	self.dataset = dataset
	self.labelset = labelset
	
	self.init_states = init_states
	self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

	self.provide_data = [('data',(batch_size, seq_len, 4096))]+init_states
	self.provide_label = [('label',(batch_size, seq_len, 1))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
	for k in range(self.count):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * self.batch_size + i
		data.append(self.dataset[idx])
		label.append(self.labelset[idx])
	
	    data_all = [mx.nd.array(data)]+self.init_state_arrays
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']+init_state_names
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':
    num_hidden = 2048
    num_lstm_layer = 5
    batch_size = BATCH_SIZE

    num_epoch = 500
    learning_rate = 0.0025
    momentum = 0.0015
    num_label = 101
    seq_len = 28

    contexts = [mx.context.gpu(0)]

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, num_hidden, num_label)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    ##data input
    (x_train, x_test) = vgg_predict()
    (y_train, y_test) = get_label()

    data_train = LRCNIter(x_train, y_train, train_data_count, batch_size, seq_len, init_states)
    data_test = LRCNIter(x_test, y_test, test_data_count, batch_size, seq_len, init_states)

    symbol = sym_gen(seq_len)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
				 num_epoch=num_epoch,
				 learning_rate=learning_rate,
				 momentum=momentum,
				 wd=0.00001,
				 initializer=mx.init.Xavier(factor_type="in",magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    debug_metrics = mx.metric.np(Accuracy)

    model.fit(X=data_train, eval_data=data_test, eval_metric=debug_metrics)
