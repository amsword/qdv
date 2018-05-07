import base64
import os.path as op
import os
import urllib
import gzip
import struct
import numpy as np
import logging
import scipy
from qd_common import ensure_directory, write_to_file
from process_tsv import tsv_writer
import cv2

def ensure_mnist_tsv():
    def gen_rows(images, labels):
        for i in range(len(labels)):
            im = cv2.imencode('.png', images[i])[1]
            image_data = base64.b64encode(im)
            yield '{}_{}'.format(i, str(labels[i])), str(labels[i]), image_data 
    target_file = './data/mnist/train.tsv'
    if not op.isfile(target_file) or True:
        tsv_writer(gen_rows(run(0), run(1)), target_file)

    target_file = './data/mnist/test.tsv'
    if not op.isfile(target_file):
        tsv_writer(gen_rows(run(2), run(3)), target_file)

    target_file = './data/mnist/labelmap.txt'
    if not op.isfile(target_file):
        write_to_file('\n'.join(map(str, range(10))), target_file)

def run(idx):
    if idx == 0: # train
        base_name = 'train-images-idx3-ubyte.gz' 
    elif idx == 1: # train label
        base_name = 'train-labels-idx1-ubyte.gz' 
    elif idx == 2: # test
        base_name = 't10k-images-idx3-ubyte.gz' 
    elif idx == 3: # test label
        base_name = 't10k-labels-idx1-ubyte.gz' 
    file_name = op.join('./data', 'mnist', base_name)
    _ensure_data_file(file_name);
    if idx == 1 or idx == 3:
        return _read_label(file_name);
    else:
        return _read_image(file_name);

def _ensure_data_file(file_name):
    if not os.path.exists(file_name):
        ensure_directory(os.path.dirname(file_name));
        base_name = os.path.basename(file_name)
        url = 'http://yann.lecun.com/exdb/mnist/' + base_name;
        urllib.urlretrieve(url, file_name)

def _read_label(file_name):
    with gzip.open(file_name, 'rb') as fp:
        magic, size = struct.unpack('>II', fp.read(8))
        assert magic == 2049, '{} is broken'.format(file_name)
        labels = np.frombuffer(fp.read(), dtype = np.uint8);
    return labels;

def _read_image(file_name):
    with gzip.open(file_name, 'rb') as fp:
        magic, size, rows, cols = struct.unpack('>IIII', fp.read(16)) 
        assert magic == 2051, '{} is broken'.format(file_name);
        imgs = np.frombuffer(fp.read(), dtype = np.uint8)
        imgs = imgs.reshape((size, rows, cols))
        return imgs

if __name__ == '__main__':
    create_mnist_tsv()
    #images = run(0);
    #labels = run(1);
    #for i in range(10):
        #im = images[i];
        #l = labels[i];
        #scipy.misc.imsave('/home/jianfw/work/mnist/{}'.format(str(l) + '.png'), im);

