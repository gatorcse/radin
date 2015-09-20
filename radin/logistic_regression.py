__author__ = 'tlohman'

import theano
from theano import tensor as T
import numpy as np
from load import mnist
import os
import time
import math

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

start_time = time.time()

##################################################
data_folder = 'truncated_data' ###################
dimension = 331 ###################################
##################################################

trX, teX, trY, teY = mnist(onehot=True, datasets_dir=os.getcwd() + '/' + data_folder + '_' + str(dimension) + '/', dimension=dimension)

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((math.pow(dimension, 2), 2))

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.0001]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(10):
    print 'iteration: ' + str(i)
    for start, end in zip(range(0, len(trX), 20), range(21, len(trX), 20)):
        print 'costing: ' + str(start) + ' : ' + str(end)
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))
    print str(predict(teX))


end_time = time.time()
print 'Processing completed in ' + str(end_time - start_time) + ' seconds.'
