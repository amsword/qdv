from qd_common import ensure_directory
from numpy import linalg as la
import numpy as np
from qd_common import write_to_file
import os.path as op
from qd_common import load_net
import caffe
from yoloinit import last_linear_layer_name
from tsv_io import TSVDataset
from qd_common import construct_model
from qd_common import init_logging
import logging

def extract_last_linear_bottom_top(net_proto):
    n_layer = len(net_proto.layer)
    for i in reversed(range(n_layer)):
        l = net_proto.layer[i]
        if l.type=='InnerProduct' or l.type=='Convolution' :
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            return l.bottom[0], l.top[0], l.name
    assert False

def min_l2_init(old_train_proto,
        old_train_net_param,
        new_train_net_proto,
        new_training_tsv,
        num_new_label,
        eps,
        init_model_path):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    old_net = caffe.Net(str(old_train_proto),
            old_train_net_param,
            caffe.TEST)

    extract_new_proto = load_net(old_train_proto)
    if extract_new_proto.layer[0].type == 'TsvData':
        extract_new_proto.layer[0].tsv_data_param.source = new_training_tsv 
    else:
        assert False

    bottom_name, top_name, layer_name = extract_last_linear_bottom_top(extract_new_proto)
    logging.info(bottom_name)
    logging.info(top_name)

    extract_new_proto_file = init_model_path + '.prototxt'
    write_to_file(str(extract_new_proto), extract_new_proto_file)

    new_net = caffe.Net(str(extract_new_proto_file), old_train_net_param, caffe.TEST)
    
    def extract_data(net, num=1000):
        all_feature, all_conf, all_label = [], [], []
        while num > 0:
            net.forward(end=top_name)
            feat = np.squeeze(net.blobs[bottom_name].data)
            conf = np.squeeze(net.blobs[top_name].data)
            label = np.squeeze(net.blobs['label'].data)
            all_feature.append(feat)
            all_conf.append(conf)
            all_label.append(label)
            num = num - len(feat)
        return all_feature, all_conf, all_label
    
    all_old_feature, all_old_conf, all_old_label = extract_data(old_net)

    alpha = 0
    
    XXT = 0
    XY = 0
    total_feature = len(all_old_feature) * len(all_old_feature[0])
    for old_feature, old_conf, old_label in zip(all_old_feature, all_old_conf, all_old_label):
        C = old_conf.shape[1]
        # predict_at_label_idx - np.mean(all_predict_except_label_idx)
        a = np.choose(old_label.astype(np.int32), old_conf.T) * C / (C - 1) - \
                np.sum(old_conf, axis=1) / (C - 1)
        alpha += np.mean(a)

        XT = np.append(old_feature, np.ones((len(old_feature), 1)), axis=1)
        y = (np.sum(old_conf, axis=1) - np.choose(old_label.astype(np.int32),
                old_conf.T)) / (C - 1)
        YT = np.tile(y, [num_new_label, 1])
        XXT += np.dot(XT.T, XT)
        XY += np.dot(XT.T, YT.T)

    alpha /= len(all_old_feature)
    assert alpha > 0, 'base model is not good enough'
    th = 1
    if alpha > th:
        alpha = th

    all_new_feature, all_new_conf, all_new_label = extract_data(new_net)
    total_feature += len(all_new_feature) * len(all_new_feature[0])

    for new_feature, new_conf, new_label in zip(all_new_feature, all_new_conf,
            all_new_label):
        negative_conf = np.mean(new_conf, axis=1)
        positive_conf = negative_conf + alpha
        Y = np.tile(negative_conf.reshape((len(negative_conf), 1)), [1, num_new_label])
        flatten_index = np.ravel_multi_index([range(len(new_label)), new_label.astype(np.int32)], 
                [Y.shape[0], Y.shape[1]])
        np.put(Y, flatten_index, positive_conf)
        XT = np.append(new_feature, np.ones((len(new_feature), 1)), axis=1)
        XXT += np.dot(XT.T, XT)
        XY += np.dot(XT.T, Y)
    eps = eps * total_feature
    Wb = la.solve(XXT + eps * np.identity(XXT.shape[0]), XY)

    target_net = caffe.Net(str(new_train_net_proto), caffe.TEST)
    target_net.copy_from(old_train_net_param, ignore_shape_mismatch=True)
    old_w = old_net.params[layer_name][0].data
    new_w = Wb.T[:, :-1]
    target_net.params[layer_name][0].data[...] = np.append(old_w, new_w, axis=0)
    old_b = old_net.params[layer_name][1].data
    new_b = Wb.T[:, -1]
    target_net.params[layer_name][1].data[...] = np.append(old_b, new_b)
    ensure_directory(op.dirname(init_model_path))
    target_net.save(init_model_path)
    logging.info('old_w: {}; new_w: {}'.format(np.mean(np.abs(old_w[:])),
        np.mean(np.abs(new_w[:]))))
    logging.info('old_b: {}; new_b: {}'.format(np.mean(np.abs(old_b[:])),
        np.mean(np.abs(new_b[:]))))

def test_min_l2_init():
    old_output = './output/cifar10_first5_vggstyle_B_bn_32_3_2_2_weightdecay0.0001_add_res/'
    old_train_net_param = op.join(old_output, 'snapshot', 'model_iter_64000.caffemodel')
    old_train_net_proto = op.join(old_output, 'train.prototxt')
    new_training_tsv = './data/cifar10_second5/train.tsv'
    init_model_path = './output/tmp/a.caffemodel'

    min_l2_init(old_train_net_proto, old_train_net_param, new_training_tsv,
            5,
            init_model_path)

if __name__ == "__main__":
    init_logging()
    test_min_l2_init()

