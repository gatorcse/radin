__author__ = 'tlohman'

import os

import theano
import theano as T
import numpy
import dicom

data_dir = os.getcwd() + '/sample_data/'
input_files = filter(lambda filename: filename.endswith(".dcm"), os.listdir(data_dir))

def read_dicom_file(filename):
    input = dicom.read_file(data_dir + filename)
    pixels = input.pixel_array
    return pixels

def shared_dataset(data_xy)

whole_set = map(read_dicom_file, input_files)

