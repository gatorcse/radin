import numpy as np
import os
import dicom
import math

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def read_dicom(filename):
    return dicom.read_file(filename).pixel_array

def mnist(ntrain=139,ntest=139,onehot=True, datasets_dir='', dimension=0):
    # data_dir = os.path.join(datasets_dir,'mnist/')
    # data_dir = datasets_dir

    filenames = map(lambda f: datasets_dir + f, os.listdir(datasets_dir))
    filenames.sort()
    input_files = filter(lambda f: f.endswith('.dcm'), filenames)
    whole_set = map(read_dicom, input_files)
    reduced = map(lambda row: np.array(row).ravel(), whole_set)
    loaded = np.array(reduced)

    trX = loaded.reshape((len(input_files), math.pow(dimension, 2))).astype(float)

    label_file = open(datasets_dir + 'label.txt', 'r').read()
    lines = label_file.splitlines()
    lines.sort()
    labels = map(lambda x: int(x.split(' ')[1]), lines)
    # loaded = np.array(labels)

    # fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # trY = loaded.reshape((len(labels)))
    trY = np.array(labels)

    # fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teX = loaded[16:].reshape((10000,28*28)).astype(float)
    #
    # fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teY = loaded[8:].reshape((10000))

    teX = trX
    teY = trY

    trX = trX/255.
    teX = teX/255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 2)
        teY = one_hot(teY, 2)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX,teX,trY,teY
