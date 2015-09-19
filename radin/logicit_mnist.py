__author__ = 'tlohman'

import os

import theano
import theano as T
import numpy
import dicom

data_dir = os.getcwd() + '/sample_data/'
inputFiles = []

for filename in os.listdir(data_dir):

    # inputFiles.append(open(filename, 'r'))
    print filename
