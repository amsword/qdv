#import matplotlib
#matplotlib.use('Agg')
import sys
sys.path.append('./scripts')
import shutil
import random
import matplotlib.pyplot as plt
import base64
import copy
import cv2
from yolodet import im_detect, result2bblist
import os.path as op
import re
from pprint import pformat
import _init_paths
from quickcaffe.modelzoo import VGGStyle
import caffe
from yolotrain import yolotrain
import multiprocessing as mp
from yolotrain import CaffeWrapper
from pprint import pprint
import simplejson as json
import os
import Queue
import time
from qd_util import BatchProcess
from tsv_io import TSVDataset, tsv_reader
from qd_common import write_to_yaml_file
from qd_common import caffe_train, write_to_file, load_net
from qd_common import default_data_path, caffe_net_check
from qd_common import get_solver, encode_expid, construct_model
from qd_common import init_logging, parse_yolo_log, parse_yolo_log_acc
from qd_common import parse_yolo_log_st
from qd_common import load_binary_net
from qd_common import read_to_buffer
from qd_common import load_from_yaml_file
from taxonomy import load_label_parent, get_nick_name, noffset_to_synset
from process_tsv import img_from_base64
from process_image import draw_bb, show_image, show_images
from latex_writer import print_table, print_m_table
from latex_writer import print_csv_table
from remote_run import remote_python_run, sync
from remote_run import sync_qd
import logging
import numpy as np
import cPickle as pkl
from yolodet import tsvdet, prepare_net_input
from deteval import deteval
from demo_detection import test_demo
from gpu_util import gpu_available
from qd_common import check_best_iou
from tsv_io import tsv_writer
from create_mnist import ensure_mnist_tsv
from yolotrain import yolotrain_main
from process_tsv import TSVFile, convert_one_label
from process_tsv import populate_dataset_details
from process_tsv import visualize_tsv2, load_labels
from process_tsv import update_confusion_matrix
from qd_common import calculate_iou
from qd_common import load_list_file
import glob
import random
from collections import OrderedDict
from qd_util import parse_test_data, dataset_op_select
from qd_util import url_to_image
from qd_util import read_blob
from process_image import network_input_to_image
from qd_util import lmdb_reader
from caffe.proto import caffe_pb2
from itertools import izip
from process_tsv import build_taxonomy_impl
from qd_util import philly_upload
from qd_util import philly_mkdir
from qd_util import philly_upload_dir
from qd_util import philly_ls
from qd_util import philly_download
from process_image import save_image
from process_image import put_text
from qd_util import update_rects_within_image
from qd_util import add_prediction_into_train
from qd_util import upload_qdoutput


def get_machine():
    # use a file to load the machines. The benefit is that sometimes the run.py
    # will be copied as anohter run_123.py, which can benefit from the new list
    machines = load_from_yaml_file('.machines.yaml')
    return machines

def get_all_resources(exclude=[], includeonly=[], num_gpu=None):
    if len(sys.argv) == 2:
        # in philly by the command of run.py num_gpu
        return [({}, range(int(float(sys.argv[-1]))))]
    machines = get_machine()
    clusters8 = machines['clusters8']
    clusters4 = machines['clusters4']
    clusters2 = machines['clusters2']
    
    all_resource = []
    def add_djx(all_resource):
        vigs = machines['vigs']
        for i, c in enumerate(vigs):
            if num_gpu is None:
                #if i == 0 or i == 1:
                for g in [[0,1,2,3], [4,5,6,7]]:
                    all_resource += [(c, g)]
                #elif i == 1:
                    #for g in [[0,1,2,3]]:
                        #all_resource += [(c, g)]

            else:
                start = 0
                while True:
                    if num_gpu + start > 8:
                        break
                    #if i == 1 and start >= 4 and start <= 7:
                        #break
                    all_resource += [(c, range(start, num_gpu + start))]
                    start = start + num_gpu

    if len(includeonly) > 0:
        assert len(exclude) == 0
        assert len(includeonly) == 1
        assert includeonly[0] == 'djx'
        add_djx(all_resource)
        return all_resource

    if 'djx' not in exclude:
        add_djx(all_resource)
    for c in clusters8:
        if num_gpu is None or num_gpu == 8:
            for g in [[0,1,2,3,4,5,6,7]]:
                all_resource += [(c, g)]
        elif num_gpu == 4:
            for g in [[0,1,2,3], [4,5,6,7]]:
                all_resource += [(c, g)]
    for c in clusters4:
        for g in [[0,1,2,3]]:
            if num_gpu is None or num_gpu == 4:
                all_resource += [(c, g)]
    for c in clusters2:
        for g in [[0,1]]:
            if num_gpu is None or num_gpu == 2:
                all_resource += [(c, g)]
    return all_resource 

def check_net():
    solver = './output/voc2_darknet19_const_lr/solver.prototxt'
    param = \
    './output/voc2_darknet19_const_lr/snapshot/model_iter_100.solverstate'
    s = get_solver(solver, param)
    x = caffe_net_check(s.net)
    logging.info('region_loss' in x['param'])
    logging.info(s.net.params['region_loss'][0].data)


def task_processor(resource, task, func=yolotrain_main):
    ssh_info, gpus = resource
    logging.info(ssh_info)
    logging.info(gpus)
    task['gpus'] = gpus
    if ssh_info == {}:
        #yolotrain_main(**task)
        func(**task)
    else:
        #yolotrain(**task)
        remote_python_run(func, task, ssh_info)

def extend_task(all_task, expid, **kwargs):

    return expid

def pipe_run():
    all_task = []
    max_num = 500
    all_task = []
    machine_ips = []
    #for monitor_train_only in [True]:
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    num_extra_convs = 3
    #suffix = '_1'
    suffix = '_2gpu'
    #suffix = '_xx'
    #dataset_ops_template = []
    #dataset_ops = []
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    batch_size_factor = 1
    suffix = '_batchSizeFactor{}'.format(batch_size_factor) \
            if batch_size_factor != 1 else ''
    #suffix = '{}{}'.format(suffix, '_256e')
    suffix = '_withNoBB' 
    suffix = '' 
    effective_batch_size = 64 * batch_size_factor
    #max_iters=1
    def gen_anchor_bias(num_anchor):
        if num_anchor == 0:
            return None
        if num_anchor == 2:
            return [4, 8, 8, 4]
        result = []
        n = int(np.sqrt(num_anchor))
        assert n * n == num_anchor
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 9
    anchor_bias = gen_anchor_bias(num_anchor)
    anchor_bias = None
    multi_feat_anchor = [{'feature': 'dark5e/leaky', 
        'extra_conv_channels': [1024],
        'extra_conv_kernels': [1],
        'anchor_bias': [1, 1], 
        'stride': 16}, 
        {'feature': 'dark6e/leaky', 
            'extra_conv_channels': [1024, 1024, 1024],
            'extra_conv_kernels': [1, 3, 1],
            'stride': 32,
            'anchor_bias': [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]}]
    multi_feat_anchor = []
    num_anchor = 5
    multibin_wh = False
    multibin_wh_low = 0;
    multibin_wh_high = 13
    all_multibin_wh_count = [16]
    #all_multibin_wh_count = [16, 32, 48, 64];
    yolo_rescore = True
    yolo_xy_kl_distance = False
    yolo_obj_only = False
    yolo_exp_linear_wh = False
    yolo_obj_nonobj_align_to_iou = False
    yolo_obj_ignore_center_around = False
    yolo_obj_kl_distance = False
    yolo_obj_set1_center_around = False
    yolo_blame = 'xy.wh.obj.cls.nonobj'
    yolo_blame = 'xy.wh.obj.nonobj.cls'
    yolo_blame = ''
    yolo_deconv_to_increase_dim = False
    yolo_deconv_to_increase_dim_adapt_bias = True
    yolo_anchor_aligned_images = 1280000000
    yolo_anchor_aligned_images = 12800
    #yolo_anchor_aligned_images = 0
    yolo_nonobj_extra_power = 0
    yolo_obj_extra_power = 0
    yolo_disable_data_augmentation = False
    yolo_disable_data_augmentation_except_shift = False
    #yolo_anchor_aligned_images = 0
    bn_no_train = False
    yolo_coords_only = False
    if yolo_coords_only:
        yolo_object_scale = 0
        yolo_noobject_scale = 0
        yolo_class_scale = 0
    else:
        yolo_object_scale = 5
        yolo_noobject_scale = 1
        yolo_class_scale = 1
    yolo_avg_replace_max = False
    yolo_per_class_obj = False
    max_iters = '128e'
    data = 'voc20'
    #data = 'brand1048'
    #data = 'office100_v1'
    if data == 'voc20':
        max_iters = 10000
    elif data == 'fridge_clean':
        max_iters = 10000
    #max_iters=10000 / batch_size_factor
    burn_in = '5e'
    burn_in = ''
    burn_in_power = 1

    yolo_full_gpu = True
    #yolo_full_gpu = False
    test_tree_cls_specific_th_by_average = 1.2
    test_tree_cls_specific_th_by_average = None
    yolo_angular_loss = True
    yolo_angular_loss = False
    no_bias = 'conf'
    no_bias = ''
    init_from = {'data': 'imagenet', 'net': 'darknet19_448', 'expid': 'A'}
    #init_from = {'data': 'office_v2.1', 'net': 'darknet19_448', 
        #'expid': 'A_burnIn5e.1_tree_initFrom.imagenet.A'}
    #init_from = {}
    max_iters = '5000e'
    kwargs_template = dict(
            detmodel='yolo',
            max_iters=max_iters,
            #max_iters='80e',
            #max_iters='256e',
            #max_iters=max_iters,
            #max_iters=10000,
            #yolo_blame=yolo_blame,
            #expid=expid,
            #yolo_jitter=0,
            #yolo_hue=0,
            #yolo_test_fix_xy = True,
            #yolo_test_fix_wh = True,
            #yolo_extract_target_prediction = True,
            #yolo_max_truth=300,
            #yolo_exposure=1,
            #test_on_train = True,
            #yolo_saturation=1,
            #yolo_random_scale_min=1,
            #yolo_random_scale_max=1,
            #expid='A_multibin_wh_0_13_16_no2-wh',
            #expid='baseline_2',
            #snapshot=1000,
            #snapshot=0,
            #target_synset_tree='./aux_data/yolo/9k.tree',
            #target_synset_tree='./data/{}/tree.txt'.format(data),
            #dataset_ops=dataset_ops,
            #effective_batch_size=1,
            #num_anchor=3,
            #num_anchor=num_anchor,
            #force_train=True,
            #force_evaluate=True,
            #debug_detect=True,
            #force_predict=True,
            #extract_features='angular_loss.softmax_loss.o_obj_loss.xy_loss.wh_loss.o_noobj_loss',
            #data_dependent_init=True,
            #restore_snapshot_iter=-1,
            #display=0,
            #region_debug_info=10,
            #display=100,
            #stagelr=stagelr,
            #anchor_bias=anchor_bias,
            #test_input_sizes=[288, 416, 480, 608],
            test_input_sizes=[2080],
            #test_input_sizes=[416, 544, 608, 992, 1024],
            #test_input_sizes=[992],
            #stageiter=[1000, 1000000],
            #stageiter=[100,5000,9000,10000000],
            #stageiter = (np.asarray([5000, 9000, 1000000]) / batch_size_factor).tolist(),
            #stagelr = (np.asarray([0.001,0.0001,0.00001]) * batch_size_factor).tolist(),
            #stagelr=[0.0001,0.001,0.0001,0.0001],
            #burn_in=100,
            #class_specific_nms=False,
            #basemodel='./output/imagenet_darknet19_448_A/snapshot/model_iter_570640.caffemodel',
            #effective_batch_size=effective_batch_size,
            #solver_debug_info=True,
            yolo_test_maintain_ratio = True,
            ovthresh = [0,0.1,0.2,0.3,0.4,0.5])

    if effective_batch_size != 64:
        kwargs_template['effective_batch_size'] = effective_batch_size

    if yolo_disable_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
        kwargs_template['yolo_fix_offset'] = True
        kwargs_template['yolo_mirror'] = False
    elif yolo_disable_data_augmentation_except_shift:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1

    continue_less_data_augmentation = False
    if continue_less_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
    #import ipdb;ipdb.set_trace()
    #if batch_size_factor == 2:
        #burn_in = '5e'
        #burn_in_power = 1

    #adv = [(False, True), (True, False), (True, True)]
    #max_iters = 4
    #for multibin_wh_count in all_multibin_wh_count:
    multibin_wh_count = all_multibin_wh_count[0]
    #for yolo_obj_ignore_center_around, yolo_obj_kl_distance in adv:
    yolo_fixed_target = False
    yolo_obj_nonobj_nopenaltyifsmallthaniou = False
    yolo_obj_cap_center_around = False
    yolo_multibin_xy = False
    yolo_sigmoid_xy = True
    yolo_delta_region3 = False
    yolo_background_class = False
    #yolo_background_class = False
    yolo_use_background_class_to_reduce_obj = 0.4
    #for multibin_xy_count in [32, 16, 8]:
    multibin_xy_count = 4
    #res_loss = True
    res_loss = False
    yolo_use_background_class_to_reduce_obj = 1
    monitor_train_only = False
    #monitor_train_only = True
    #for yolo_use_background_class_to_reduce_obj in [1, 0.8, 0.6, 0.4, 0.2]:
    dataset = TSVDataset(data)
    yolo_low_shot_regularizer = False
    #full_labels = dataset.load_labelmap()
    full_labels = []
    #for low_shot_label in full_labels:
    yolo_random_scale_max = 2
    scale_relative_input = True
    scale_relative_input = False
    nms_type = 'LinearSoft'
    nms_type = 'GaussianSoft'
    nms_type = ''
    gaussian_nms_sigma = 0.5
    scale_constrained_by_one_box_area_min = 0.001
    last_fixed_param = 'dark5e/leaky'
    last_fixed_param = ''
    #for low_shot_label in ['']:
    #for low_shot_label in [full_labels[0]]:
    full_labels.insert(0, '')
    #for low_shot_label in full_labels[:5]:
    low_shot_label = ''
    residual_loss = True
    residual_loss = False
    residual_loss_froms = ['extra_conv19/leaky', 'extra_conv20/leaky']
    #for low_shot_label in ['']:
    #for extra_conv_kernel in [[1, 1], [1, 3], [3, 1]]:
    yolo_disable_no_penalize_if_iou_large = False
    cutout_prob = -1
    expid_prefix = 'A'
    yolo_random_scale_min = 2
    yolo_random_scale_max = 4
    net_input_size_min = 416 * 2
    net_input_size_max = 416 * 2
    #yolo_random_scale_min = 0.25
    #yolo_random_scale_max = 2
    for extra_conv_kernel in [[3, 3, 3]]:
    #for extra_conv_channels in [[1024, 512, 512], [512, 512, 1024], [512, 1024,
        #512]]:
        extra_conv_channels = [1024, 1024, 1024]
    #for extra_conv_channels in [[1024, 1024]]:
    #for extra_conv_channels in [[1024, 1024, 1024]]:
    #for extra_conv_kernel in [[3,3,3], [1, 1, 1], [1, 3, 1], [3, 1, 1], [1, 1, 3], [3,
        #3, 1], [3, 1, 3], [1, 3, 3]]:
    #for extra_conv_kernel in [[1, 3, 1]]:
        if len(low_shot_label) > 0:
            dataset_ops = [{'op':'low_shot', 'labels': 'dog', 'num_train': 1}]
            dataset_ops[0]['labels'] = low_shot_label
            dataset_ops[0]['labels_idx'] = full_labels.index(low_shot_label)
        else:
            dataset_ops = [{'op':'remove'},
                    {'op':'add',
                     'name':'office_v2.11_with_bb',
                     'source':'train',
                     'weight': 3},
                    #{'op': 'add',
                     #'name': 'office_v2.1_no_bb',
                     #'source': 'train',
                     #'weight': 1},
                    ]
            dataset_ops = []
            #for multibin_xy_count in [4]:
        yolo_multibin_xy_low = 0.5 / multibin_xy_count
        #multibin_xy_count = 16
        yolo_multibin_xy_high = 1 - 0.5 / multibin_xy_count
        yolo_multibin_xy_count = multibin_xy_count
        #for yolo_obj_kl_distance in [False]:
        yolo_obj_kl_distance = False
    #for yolo_object_scale in [40, 60]:
        #for monitor_train_only in [False, True]:
        #for monitor_train_only in [True]:
        #for yolo_iou_th_to_use_bkg_cls in [0.1]:
        yolo_iou_th_to_use_bkg_cls = 0.1
        #for last_conv_bias in [True]:
        last_conv_bias = True
        #for rotate_max in [5, 10, 15, 20]:
        rotate_max = 10
        rotate_max = 0
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, True), (True, False)]:
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, False)]:
        incorporate_at_least_one_box, scale_constrained_by_one_box_area = False, False
        #for yolo_angular_loss_weight in [0.1, 1, 10]:
        #for yolo_angular_loss_weight in [1]:
        yolo_angular_loss_weight = 1
        #for data in ['fridge_clean', 'voc20']:
        #for data in ['office_v2.1']:
        for data in ['pipe']:
        #for data in ['coco_phone']:
        #for data in ['coco2017']:
            if data.startswith('office'):
                yolo_tree = True
            else:
                yolo_tree = False
            if test_tree_cls_specific_th_by_average:
                assert yolo_tree
        #rotate_max = 10
        #rotate_max = 0
            rotate_with_90 = False
        #for yolo_random_scale_min in [0.5, 0.75]:
        #for monitor_train_only in [False]:
        #monitor_train_only = True
        #monitor_train_only = True
    #for monitor_train_only in [True]:
    #for monitor_train_only in [False, True]:
            #for add_reorg, num_extra_convs in [(True, [6]), (True, [7])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(False, [3]), (True, [3])]:
            for add_reorg, num_extra_convs in [(True, [3])]:
        #for add_reorg, num_extra_convs in all_extra_option:
        #for weight_decay in [0.0001, 0.00005]:
        #for weight_decay in [0.0005]:
            #weight_decay = 0.0005
            #weight_decay = 0
            #for net in ['darknet19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
            #for net in ['resnet34']:
            #net = 'darknet19'
            #for net in ['resnet101']:
            #for yolo_coord_scale in [1]:
                yolo_coord_scale = 1
                #for data in ['voc20', 'voc2012', 'coco2017', 'imagenet']:
                #for data in ['coco2017']:
                #for weight_decay in [0.001, 0.005, 0.01, 0.05]:
                weight_decay = 0.0005
                #for net in ['resnet34', 'darknet19_448']:
                #for net in ['darknet19_448', 'resnet34']:
                for net in ['darknet19_448']:
                #for net in ['resnet101']:
                #for net in ['darknet19_448', 'resnet34', 'resnet101']:
                    if len(burn_in) > 0 and data != 'voc20':
                        kwargs_template['stageiter'] = ['60e', '90e', '900e'] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()
                    elif len(burn_in) > 0:
                        kwargs_template['stageiter'] = [5000, 9000, 100000] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()

                    kwargs = copy.deepcopy(kwargs_template)
                    expid = expid_prefix
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if len(num_extra_convs) == 0 and num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                        kwargs['num_extra_convs'] = num_extra_convs
                    elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3 \
                            or len(num_extra_convs) > 1:
                        expid = expid + '_extraconv{}'.format(
                                '_'.join(map(str, num_extra_convs)))
                        kwargs['num_extra_convs'] = num_extra_convs
                    if multibin_wh:
                        expid = expid + '_multibin_wh_{}_{}_{}'.format(multibin_wh_low,
                                multibin_wh_high, multibin_wh_count)
                        kwargs['multibin_wh'] = multibin_wh
                        kwargs['multibin_wh_low'] = multibin_wh_low
                        kwargs['multibin_wh_high'] = multibin_wh_high
                        kwargs['multibin_wh_count'] = multibin_wh_count
                    if not add_reorg:
                        kwargs['add_reorg'] = add_reorg
                    if num_anchor != 5:
                        expid = '{}_numAnchor{}'.format(expid, num_anchor)
                        assert len(anchor_bias) == 2 * num_anchor
                        kwargs['anchor_bias'] = anchor_bias
                    if not yolo_rescore:
                        expid = '{}_{}'.format(expid, 'norescore')
                        kwargs['yolo_rescore'] = yolo_rescore
                    if yolo_obj_ignore_center_around:
                        expid = '{}_{}'.format(expid, 'ignore')
                        kwargs['yolo_obj_ignore_center_around'] = yolo_obj_ignore_center_around
                    if yolo_obj_kl_distance:
                        expid = '{}_{}'.format(expid, 'objkl')
                        kwargs['yolo_obj_kl_distance'] = yolo_obj_kl_distance
                    if yolo_xy_kl_distance:
                        expid = '{}_xykl'.format(expid)
                        kwargs['yolo_xy_kl_distance'] = yolo_xy_kl_distance
                    if yolo_obj_only:
                        expid = '{}_objonly'.format(expid)
                        kwargs['yolo_obj_only'] = yolo_obj_only
                    if yolo_exp_linear_wh:
                        expid = '{}_explinearwh'.format(expid)
                        kwargs['yolo_exp_linear_wh'] = yolo_exp_linear_wh
                    if yolo_obj_nonobj_align_to_iou:
                        expid = '{}_nonobjtoiou'.format(expid)
                        kwargs['yolo_obj_nonobj_align_to_iou'] = yolo_obj_nonobj_align_to_iou
                    if yolo_obj_set1_center_around:
                        expid = '{}_around1'.format(expid)
                        kwargs['yolo_obj_set1_center_around'] = yolo_obj_set1_center_around
                    if yolo_obj_nonobj_nopenaltyifsmallthaniou:
                        expid = '{}_nolosssmallthaniou'.format(expid)
                        kwargs['yolo_obj_nonobj_nopenaltyifsmallthaniou'] = yolo_obj_nonobj_nopenaltyifsmallthaniou
                    if yolo_obj_cap_center_around:
                        expid = '{}_objcaparound'.format(expid)
                        kwargs['yolo_obj_cap_center_around'] = yolo_obj_cap_center_around
                    if weight_decay != 0.0005:
                        expid = '{}_decay{}'.format(expid, weight_decay)
                        kwargs['weight_decay'] = weight_decay
                    if yolo_fixed_target:
                        expid = '{}_fixedtarget'.format(expid)
                        kwargs['yolo_fixed_target'] = yolo_fixed_target
                    if yolo_deconv_to_increase_dim:
                        expid = '{}_deconvincreasedim'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim'] = True
                    if yolo_coord_scale != 1:
                        expid = '{}_coordscale{}'.format(expid,
                                yolo_coord_scale)
                        kwargs['yolo_coord_scale'] = yolo_coord_scale
                    if not yolo_deconv_to_increase_dim_adapt_bias:
                        expid = '{}_nobiaseadapt'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim_adapt_bias'] = \
                            False
                    if yolo_anchor_aligned_images != 12800:
                        expid = '{}_align{}'.format(expid,
                                yolo_anchor_aligned_images)
                        kwargs['yolo_anchor_aligned_images'] = yolo_anchor_aligned_images
                    if yolo_nonobj_extra_power != 0:
                        expid = '{}_nonobjpower{}'.format(expid,
                                yolo_nonobj_extra_power)
                        kwargs['yolo_nonobj_extra_power'] = yolo_nonobj_extra_power
                    if yolo_obj_extra_power != 0:
                        expid = '{}_objpower{}'.format(expid,
                                yolo_obj_extra_power)
                        kwargs['yolo_obj_extra_power'] = yolo_obj_extra_power
                    if yolo_multibin_xy:
                        kwargs['yolo_multibin_xy'] = yolo_multibin_xy
                        kwargs['yolo_multibin_xy_low'] = yolo_multibin_xy_low
                        kwargs['yolo_multibin_xy_high'] = yolo_multibin_xy_high
                        kwargs['yolo_multibin_xy_count'] = yolo_multibin_xy_count
                        expid = '{}_multibinXY{}'.format(expid,
                                yolo_multibin_xy_count)
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'select_top':
                        expid = '{}_selectTop{}'.format(expid,
                                dataset_ops[0]['num_top'])
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'low_shot':
                        low_shot_labels = dataset_ops[0]['labels']
                        low_shot_num_train = dataset_ops[0]['num_train']
                        expid = '{}_lowShot.{}.{}'.format(expid, 
                                low_shot_labels,
                                low_shot_num_train)
                    if len(dataset_ops) > 0:
                        kwargs['dataset_ops'] = dataset_ops
                    if yolo_disable_data_augmentation:
                        expid = '{}_noAugmentation'.format(expid)
                    if yolo_disable_data_augmentation_except_shift:
                        expid = '{}_noAugExpShift'.format(expid)
                    if bn_no_train:
                        expid = '{}_bnNoTrain'.format(expid)
                        kwargs['bn_no_train'] = bn_no_train
                    if yolo_object_scale != 5:
                        expid = '{}_objscale{}'.format(expid, yolo_object_scale)
                        kwargs['yolo_object_scale'] = yolo_object_scale
                    if yolo_noobject_scale != 1:
                        expid = '{}_noobjScale{}'.format(expid,
                                yolo_noobject_scale)
                        kwargs['yolo_noobject_scale'] = yolo_noobject_scale
                    if yolo_class_scale != 1:
                        expid = '{}_clsScale{}'.format(expid,
                                yolo_class_scale)
                        kwargs['yolo_class_scale'] = yolo_class_scale
                    if yolo_avg_replace_max:
                        expid = '{}_avgReplaceMax'.format(expid)
                        kwargs['yolo_avg_replace_max'] = yolo_avg_replace_max
                    if not yolo_sigmoid_xy:
                        expid = '{}_nosigmoidXY'.format(expid)
                        kwargs['yolo_sigmoid_xy'] = yolo_sigmoid_xy
                    if yolo_delta_region3:
                        expid = '{}_deltaRegion3'.format(expid)
                        kwargs['yolo_delta_region3'] = yolo_delta_region3
                    if yolo_background_class:
                        expid = '{}_bkgCls{}'.format(expid,
                                yolo_use_background_class_to_reduce_obj)
                        kwargs['yolo_background_class'] = True
                        kwargs['yolo_use_background_class_to_reduce_obj'] = yolo_use_background_class_to_reduce_obj
                        if yolo_iou_th_to_use_bkg_cls != 1:
                            expid = '{}_iouTh{}'.format(expid,
                                    yolo_iou_th_to_use_bkg_cls)
                            kwargs['yolo_iou_th_to_use_bkg_cls'] = yolo_iou_th_to_use_bkg_cls
                    if res_loss:
                        expid = '{}_resLoss'.format(expid)
                        kwargs['res_loss'] = res_loss
                        kwargs['skip_genprototxt'] = True
                    if yolo_per_class_obj:
                        expid = '{}_perClassObj'.format(expid)
                        kwargs['yolo_per_class_obj'] = yolo_per_class_obj
                    if not last_conv_bias:
                        expid = '{}_noBiasLastConv'.format(expid)
                        kwargs['yolo_last_conv_bias'] = last_conv_bias
                    if yolo_low_shot_regularizer:
                        expid = '{}_lowShotEqualNorm'.format(expid)
                        kwargs['yolo_low_shot_regularizer'] = True
                    if yolo_full_gpu:
                        expid = '{}_fullGpu'.format(expid)
                        kwargs['yolo_full_gpu'] = yolo_full_gpu
                    if burn_in != '':
                        expid = '{}_burnIn{}.{}'.format(expid, burn_in,
                                burn_in_power)
                        kwargs['burn_in'] = burn_in
                        kwargs['burn_in_power'] = burn_in_power
                    if rotate_max != 0:
                        kwargs['rotate_max'] = rotate_max
                        expid = '{}_rotate{}'.format(expid, rotate_max)
                    if rotate_with_90:
                        expid = '{}_with90'.format(expid)
                        kwargs['rotate_with_90'] = True
                    if yolo_random_scale_min != 0.25:
                        expid = '{}_randomScaleMin{}'.format(expid, 
                                yolo_random_scale_min)
                        kwargs['yolo_random_scale_min'] = yolo_random_scale_min
                    if yolo_random_scale_max != 2:
                        expid = '{}_randomScaleMax{}'.format(expid,
                                yolo_random_scale_max)
                        kwargs['yolo_random_scale_max'] = yolo_random_scale_max
                    if scale_relative_input:
                        expid = '{}_RelativeScale2'.format(expid)
                        kwargs['scale_relative_input'] = scale_relative_input
                    if nms_type != 'Standard' and nms_type != '':
                        if nms_type == 'LinearSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.LinearSoft 
                        if nms_type == 'GaussianSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.GaussianSoft
                        if gaussian_nms_sigma != 0.5:
                            kwargs['gaussian_nms_sigma'] = 0.5
                    if incorporate_at_least_one_box:
                        expid = '{}_atLeastOneBB'.format(expid)
                        kwargs['incorporate_at_least_one_box'] = incorporate_at_least_one_box
                    if scale_constrained_by_one_box_area:
                        expid = '{}_scaleConstrainedByOne'.format(expid)
                        kwargs['scale_constrained_by_one_box_area'] = scale_constrained_by_one_box_area
                        if scale_constrained_by_one_box_area_min != 0.001:
                            expid = '{}_Min{}'.format(expid, scale_constrained_by_one_box_area_min)
                            kwargs['scale_constrained_by_one_box_area_min'] = scale_constrained_by_one_box_area_min
                    if yolo_tree:
                        expid = '{}_tree'.format(expid)
                        kwargs['yolo_tree'] = yolo_tree
                        if test_tree_cls_specific_th_by_average is not None:
                            kwargs['test_tree_cls_specific_th_by_average'] = test_tree_cls_specific_th_by_average
                    if len(init_from) > 0:
                        assert net == init_from['net']
                        if len(init_from['expid']) > 5:
                            expid = '{}_initFrom'.format(expid)
                        else:
                            expid = '{}_initFrom.{}.{}'.format(expid,
                                    init_from['data'], init_from['expid'])
                        c = CaffeWrapper(data=init_from['data'], 
                                net=init_from['net'],
                                expid=init_from['expid'])
                        kwargs['basemodel'] = c.best_model().model_param
                    if yolo_angular_loss:
                        expid = '{}_AngularRegulizer'.format(expid)
                        kwargs['yolo_angular_loss'] = True
                        if yolo_angular_loss_weight != 1:
                            expid = '{}Weight{}'.format(expid,
                                    yolo_angular_loss_weight)
                            kwargs['yolo_angular_loss_weight'] = yolo_angular_loss_weight
                    if len(no_bias) > 0:
                        expid = '{}_noBias{}'.format(expid, no_bias)
                        kwargs['no_bias'] = no_bias
                    if net_input_size_min != 416:
                        expid = '{}_InMin{}'.format(expid, net_input_size_min)
                        kwargs['net_input_size_min'] = net_input_size_min
                    if net_input_size_max != 416:
                        expid = '{}_InMax{}'.format(expid, net_input_size_max)
                        kwargs['net_input_size_max'] = net_input_size_max
                    if any(k != 3 for k in extra_conv_kernel):
                        expid = '{}_extraConvKernel.{}'.format(expid, '.'.join(
                            map(str, extra_conv_kernel)))
                        kwargs['extra_conv_kernel'] = extra_conv_kernel
                    if any(c != 1024 for c in extra_conv_channels):
                        expid = '{}_extraChannels.{}'.format(expid,
                                '.'.join(map(str, extra_conv_channels)))
                        kwargs['extra_conv_channels'] = extra_conv_channels
                    if len(last_fixed_param) > 0:
                        expid = '{}_FixParam.{}'.format(expid,
                                last_fixed_param.replace('/', '.'))
                        kwargs['last_fixed_param'] = last_fixed_param
                    if data.startswith('office_v2.1'):
                       assert 'taxonomy_folder' not in kwargs
                       kwargs['taxonomy_folder'] = \
                            './aux_data/taxonomy10k/office/{}'.format(data)
                    if residual_loss:
                        expid = '{}_resLoss{}'.format(expid, 
                                '.'.join(map(lambda x: x.replace('/', '_'),
                                    residual_loss_froms)))
                        kwargs['residual_loss'] = residual_loss
                        kwargs['residual_loss_froms'] = residual_loss_froms
                    if yolo_disable_no_penalize_if_iou_large:
                        kwargs['yolo_disable_no_penalize_if_iou_large'] = True
                        expid = '{}_disableNoPenIfIouLarge'.format(expid)
                    if cutout_prob > 0:
                        kwargs['cutout_prob'] = cutout_prob
                        expid = '{}_cutoutProb{}'.format(expid, cutout_prob)
                    if len(multi_feat_anchor) > 0:
                        kwargs['multi_feat_anchor'] = multi_feat_anchor
                        expid = '{}_multiFeatAnchor{}'.format(expid,
                                len(multi_feat_anchor))
                    if max_iters != '128e' and max_iters != 10000:
                        expid = '{}_maxIter.{}'.format(expid, max_iters)
                    expid = expid + suffix
                    kwargs['monitor_train_only'] = monitor_train_only
                    kwargs['expid'] = expid
                    kwargs['net'] = net
                    kwargs['data'] = data
                    all_task.append(kwargs)

    logging.info(pformat(all_task))
    #all_gpus = [[4,5,6,7], [0,1,2,3]]
    all_gpus = [[0,1,2,3], [4,5,6,7]]
    #all_gpus = [[4,5,6,7]]
    #all_gpus = [[0,1]]
    #all_gpus = [[0]]
    #all_gpus = [[0,1,2,3]]
    #all_gpus = [[4, 5, 6, 7]]
    #all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[0]]
    vigs, clusters8, clusters4 = get_machine()
    all_resource = []
    #all_resource += [(vig2, r) for r in all_gpus]
    #all_resource += [(cluster8, r) for r in all_gpus]
    #all_resource += [(cluster8_2, r) for r in all_gpus]
    #all_resource += [(vigs[0], [0,1,2,3])]
    #all_resource += [(vigs[1], [4,5,6,7])]
    #all_resource += [(vigs[1], [0, 1, 2, 3, 4,5,6,7])]
    all_resource += [(vigs[0], [0, 1, 2, 3, 4,5,6,7])]
    #import ipdb;ipdb.set_trace()
    #all_resource += [(vigs[0], [4,5,6,7])]
    #all_resource += [(vigs[0], [0,1,2,3])]
    #for c in clusters8:
        #for g in [[0,1,2,3,4,5,6,7]]:
            #all_resource += [(c, g)]
    #for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
            #all_resource += [(c, g)]
    #for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
            #all_resource += [(c, g)]
    #all_resource += [(vigs[1], [0, 1, 2, 3, 4,5,6,7] * 7)]
    #all_resource += [(vigs[1], [4])]
    #if batch_size_factor == 2:
        #all_resource = []
        #all_resource += [(vigs[0], [0,1,2,3,4,5,6,7])]
        #all_resource += [(vigs[1], [0,1,2,3,4,5,6,7])]
        #all_resource += [(clusters8[0], [0,1,2,3,4,5,6,7])]
    #all_resource += [(clusters8[0], [0,1,2,3])]
    #all_resource += [(clusters8[0], [0,1,2,3,4,5,6,7])]
    #all_resource += [(vigs[1], [0,1,2,3,4,5,6,7])]
    #all_resource += [(cluster8_2, [0,1,2,3])]
    #all_resource += [(cluster8_2, [4,5,6,7])]
    #all_resource += [(cluster4, r) for r in all_gpus]
    #all_resource += [(cluster2_1, [0, 1])]
    #all_resource += [(cluster2_2, [0, 1])]
    logging.info(pformat(all_resource))
    logging.info('#resource: {}'.format(len(all_resource)))
    logging.info('#task: {}'.format(len(all_task)))
    debug = True
    #all_task[0]['force_predict'] = True
    debug = False
    #return
    #tsv_file = './data/office_v2.1_with_bb/test.tsv'
    #all_task[0]['force_predict'] = True
    #task = all_task[0]
    #task['expid'] = '{}_bb_nobb'.format(task['expid'])
    ##task['expid'] = '{}_bb_only'.format(task['expid'])
    #task['class_specific_nms'] = False
    #task['yolo_test_thresh'] = 0.5
    #c = CaffeWrapper(**task)
    #c.demo(None)
    #rows = tsv_reader(tsv_file)
    #for row in rows:
        #continue
    #import ipdb;ipdb.set_trace()
    #c.demo('./data/office100_v1_with_bb/train.tsv')
    #c.demo(tsv_file)
    #c.demo('/raid/jianfw/data/office100_crawl/TermList.instagram.pinterest.scrapping.image.tsv')
    #c.demo('/raid/jianfw/work/yuxiao_crop/ring/')
    #c.demo('tmp.png')
    #all_task[0]['force_predict'] = True
    #return
    def batch_run():
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
        #if not monitor_train_only:
            #for t in all_task:
                #t['monitor_train_only'] = True
            #for i, r in enumerate(all_resource):
                #all_resource[i] = (r[0], [-1] * 4)
            #b = BatchProcess(all_resource, all_task, task_processor)
            ##b._availability_check = False
            #b.run()
    if debug:
        idx = -1
        task = all_task[idx]
        task['effective_batch_size'] = 1
        #task['use_pretrained'] = False
        #all_task[idx]['max_iters'] = 1
        #task['expid'] = '{}_debug'.format(expid)
        #all_task[idx]['datas'] = ['voc20', 'crawl_office_v1']
        #all_task[idx]['force_train'] = True
        task['debug_train'] = True
        #all_task[idx]['debug_detect'] = True
        #all_task[idx]['force_predict'] = True
        #task_processor(({}, [0]), all_task[idx])
        #task['force_evaluate'] = True
        task_processor(({}, [0]), task)
        #task_processor(all_resource[-1], task)
        #task_processor((vig[1], [0]), task)
        #import ipdb;ipdb.set_trace()
    else:
        batch_run()
        pass

def yolo_incomplete_label():
    all_task = []
    max_num = 500
    all_task = []
    machine_ips = []
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    num_extra_convs = 3
    #suffix = '_1'
    suffix = '_2gpu'
    #suffix = '_xx'
    #dataset_ops_template = []
    #dataset_ops = []
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    batch_size_factor = 1
    suffix = '_batchSizeFactor{}'.format(batch_size_factor) \
            if batch_size_factor != 1 else ''
    #suffix = '{}{}'.format(suffix, '_256e')
    suffix = '_withNoBB' 
    suffix = '' 
    effective_batch_size = 64 * batch_size_factor
    #max_iters=1
    def gen_anchor_bias(num_anchor):
        if num_anchor == 0:
            return None
        if num_anchor == 2:
            return [4, 8, 8, 4]
        result = []
        n = int(np.sqrt(num_anchor))
        assert n * n == num_anchor
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 9
    anchor_bias = gen_anchor_bias(num_anchor)
    anchor_bias = None
    yolo_softmax_norm_by_valid = True
    #yolo_softmax_norm_by_valid = False
    multi_feat_anchor = [
            {   
                'feature': 'dark4c/leaky', 
                'extra_conv_channels': [512, 1024],
                'extra_conv_kernels': [3, 3],
                'loss_weight_multiplier': 1,
                'anchor_bias': [1, 1, 1, 2, 2, 1, 2, 2], 
                #'stride': 8
            },
            {
                'feature': 'dark5e/leaky', 
                'extra_conv_channels': [1024],
                'extra_conv_kernels': [3],
                #'anchor_bias': [4, 4, 2, 4, 4, 2], 
                'loss_weight_multiplier': 1,
                'anchor_bias': [2, 2, 1, 2, 2, 1], 
                #'stride': 16
            }, 
            {   
                'feature': 'dark6e/leaky', 
                'extra_conv_channels': [1024, 1024, 1024],
                'extra_conv_kernels': [3, 3, 3],
                'loss_weight_multiplier': 1,
                'anchor_bias': [2, 2, 4, 4, 4, 8, 8, 4, 8, 8],
            }
            ]
    #multi_feat_anchor = [
            #{   
                #'feature': 'dark6e/leaky', 
                #'extra_conv_channels': [1024, 1024, 1024],
                #'extra_conv_kernels': [3, 3, 3],
                #'loss_weight_multiplier': 1,
                ##'stride': 32,
                ##'anchor_bias': [6.63,11.38,9.42,5.11,16.62,10.52]
                #'anchor_bias': [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
                ##'anchor_bias': [4, 4, 2, 4, 4, 2]
            #},
            #]
    multi_feat_anchor = []
    num_anchor = 5
    multibin_wh = False
    multibin_wh_low = 0;
    multibin_wh_high = 13
    all_multibin_wh_count = [16]
    #all_multibin_wh_count = [16, 32, 48, 64];
    yolo_rescore = True
    yolo_xy_kl_distance = False
    yolo_obj_only = False
    yolo_exp_linear_wh = False
    yolo_obj_nonobj_align_to_iou = False
    yolo_obj_ignore_center_around = False
    yolo_obj_kl_distance = False
    yolo_obj_set1_center_around = False
    yolo_blame = 'xy.wh.obj.cls.nonobj'
    yolo_blame = 'xy.wh.obj.nonobj.cls'
    yolo_blame = ''
    yolo_deconv_to_increase_dim = False
    yolo_deconv_to_increase_dim_adapt_bias = True
    yolo_anchor_aligned_images = 1280000000
    yolo_anchor_aligned_images = 12800
    #yolo_anchor_aligned_images = 0
    yolo_nonobj_extra_power = 0
    yolo_obj_extra_power = 0
    yolo_disable_data_augmentation = False
    yolo_disable_data_augmentation_except_shift = False
    #yolo_anchor_aligned_images = 0
    bn_no_train = False
    yolo_coords_only = False
    if yolo_coords_only:
        yolo_object_scale = 0
        yolo_noobject_scale = 0
        yolo_class_scale = 0
    else:
        yolo_object_scale = 5
        yolo_noobject_scale = 1
        yolo_class_scale = 1
    yolo_avg_replace_max = False
    yolo_per_class_obj = False
    max_iters = '128e'
    data = 'voc20'
    #data = 'brand1048'
    #data = 'office100_v1'
    if data == 'voc20':
        max_iters = 10000
    elif data == 'fridge_clean':
        max_iters = 10000
    #max_iters=10000 / batch_size_factor
    burn_in = '5e'
    burn_in = ''
    burn_in_power = 1

    yolo_full_gpu = True
    #yolo_full_gpu = False
    test_tree_cls_specific_th_by_average = 1.2
    test_tree_cls_specific_th_by_average = None
    yolo_angular_loss = True
    yolo_angular_loss = False
    no_bias = 'conf'
    no_bias = ''
    #init_from = {'data': 'CARPK_select.5.5.nokeep', 
            #'net': 'darknet19_448', 
            #'expid': 'A_dataop9450_fullGpu_randomScaleMin2_randomScaleMax4_softmaxByValid_softmaxWeight0.2_ignoreNegativeFirst_notIgnore12800'}
    init_from = {'data': 'CARPK_select.5.5.nokeep_R905_NMS0.2', 
            'net': 'darknet19_448', 
            'expid': 'R90'}
    #init_from = {'data': 'office_v2.1', 'net': 'darknet19_448', 
        #'expid': 'A_burnIn5e.1_tree_initFrom.imagenet.A'}
    #init_from = {}
    max_iters = '128e'
    max_iters = 10000
    kwargs_template = dict(
            detmodel='yolo',
            max_iters=max_iters,
            test_data='CARPK',
            #max_iters='80e',
            #max_iters='256e',
            #max_iters=max_iters,
            #max_iters=10000,
            #yolo_blame=yolo_blame,
            #expid=expid,
            #yolo_jitter=0,
            #yolo_hue=0,
            #yolo_test_fix_xy = True,
            #yolo_test_fix_wh = True,
            #yolo_extract_target_prediction = True,
            #yolo_max_truth=300,
            #yolo_exposure=1,
            #test_on_train = True,
            #yolo_saturation=1,
            #yolo_random_scale_min=1,
            #yolo_random_scale_max=1,
            #expid='A_multibin_wh_0_13_16_no2-wh',
            #expid='baseline_2',
            #snapshot=1000,
            #snapshot=0,
            #target_synset_tree='./aux_data/yolo/9k.tree',
            #target_synset_tree='./data/{}/tree.txt'.format(data),
            #dataset_ops=dataset_ops,
            #effective_batch_size=1,
            #num_anchor=3,
            #num_anchor=num_anchor,
            #force_train=True,
            #force_evaluate=True,
            #debug_detect=True,
            #force_predict=True,
            #extract_features='angular_loss.softmax_loss.o_obj_loss.xy_loss.wh_loss.o_noobj_loss',
            #data_dependent_init=True,
            #restore_snapshot_iter=-1,
            #display=0,
            #region_debug_info=10,
            #display=100,
            #stagelr=stagelr,
            #anchor_bias=anchor_bias,
            #test_input_sizes=[288, 416, 480, 608],
            #test_input_sizes=[2080],
            #test_input_sizes=[416, 544, 608, 992, 1024],
            #test_input_sizes=[992],
            test_input_sizes=[416 * 3],
            #stageiter=[1000, 1000000],
            #stageiter=[100,5000,9000,10000000],
            #stageiter = (np.asarray([5000, 9000, 1000000]) / batch_size_factor).tolist(),
            #stagelr = (np.asarray([0.001,0.0001,0.00001]) * batch_size_factor).tolist(),
            #stagelr=[0.0001,0.001,0.0001,0.0001],
            #burn_in=100,
            #class_specific_nms=False,
            #basemodel='./output/imagenet_darknet19_448_A/snapshot/model_iter_570640.caffemodel',
            #effective_batch_size=effective_batch_size,
            #solver_debug_info=True,
            #yolo_test_maintain_ratio = True,
            ovthresh = [0,0.1,0.2,0.3,0.4,0.5])

    if effective_batch_size != 64:
        kwargs_template['effective_batch_size'] = effective_batch_size

    if yolo_disable_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
        kwargs_template['yolo_fix_offset'] = True
        kwargs_template['yolo_mirror'] = False
    elif yolo_disable_data_augmentation_except_shift:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1

    continue_less_data_augmentation = False
    if continue_less_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
    #import ipdb;ipdb.set_trace()
    #if batch_size_factor == 2:
        #burn_in = '5e'
        #burn_in_power = 1

    #adv = [(False, True), (True, False), (True, True)]
    #max_iters = 4
    #for multibin_wh_count in all_multibin_wh_count:
    multibin_wh_count = all_multibin_wh_count[0]
    #for yolo_obj_ignore_center_around, yolo_obj_kl_distance in adv:
    yolo_fixed_target = False
    yolo_obj_nonobj_nopenaltyifsmallthaniou = False
    yolo_obj_cap_center_around = False
    yolo_multibin_xy = False
    yolo_sigmoid_xy = True
    yolo_delta_region3 = False
    yolo_background_class = False
    #yolo_background_class = False
    yolo_use_background_class_to_reduce_obj = 0.4
    #for multibin_xy_count in [32, 16, 8]:
    multibin_xy_count = 4
    #res_loss = True
    res_loss = False
    yolo_use_background_class_to_reduce_obj = 1
    monitor_train_only = False
    monitor_train_only = True
    #for yolo_use_background_class_to_reduce_obj in [1, 0.8, 0.6, 0.4, 0.2]:
    dataset = TSVDataset(data)
    yolo_low_shot_regularizer = False
    #full_labels = dataset.load_labelmap()
    full_labels = []
    #for low_shot_label in full_labels:
    yolo_random_scale_max = 2
    scale_relative_input = True
    scale_relative_input = False
    nms_type = 'LinearSoft'
    nms_type = 'GaussianSoft'
    nms_type = ''
    gaussian_nms_sigma = 0.5
    scale_constrained_by_one_box_area_min = 0.001
    last_fixed_param = 'dark5e/leaky'
    last_fixed_param = ''
    #for low_shot_label in ['']:
    #for low_shot_label in [full_labels[0]]:
    full_labels.insert(0, '')
    #for low_shot_label in full_labels[:5]:
    low_shot_label = ''
    residual_loss = True
    residual_loss = False
    residual_loss_froms = ['extra_conv19/leaky', 'extra_conv20/leaky']
    #for low_shot_label in ['']:
    #for extra_conv_kernel in [[1, 1], [1, 3], [3, 1]]:
    yolo_disable_no_penalize_if_iou_large = False
    cutout_prob = 0.3
    cutout_prob = -1
    expid_prefix = 'A'
    #expid_prefix = 'B'
    #yolo_random_scale_min = 0.25
    #yolo_random_scale_max = 2
    yolo_random_scale_min = 2
    yolo_random_scale_max = 4
    net_input_size_min = 416 * 2
    net_input_size_max = 416 * 2
    net_input_size_min = 416
    net_input_size_max = 416
    #yolo_softmax_extra_weight = 0.2
    #yolo_softmax_extra_weight = 1
    ignore_negative_first_batch = True
    #yolo_not_ignore_negative_seen_images = 12800
    yolo_not_ignore_negative_seen_images = 0
    yolo_force_negative_with_partial_overlap = False
    yolo_nms = 0.2
    first_batch_objectiveness_enhancement = False
    first_batch_objectiveness_enhancement_weight = 10
    for extra_conv_kernel in [[3, 3, 3]]:
    #for extra_conv_channels in [[1024, 512, 512], [512, 512, 1024], [512, 1024,
        #512]]:
        extra_conv_channels = [1024, 1024, 1024]
    #for extra_conv_channels in [[1024, 1024]]:
    #for extra_conv_channels in [[1024, 1024, 1024]]:
    #for extra_conv_kernel in [[3,3,3], [1, 1, 1], [1, 3, 1], [3, 1, 1], [1, 1, 3], [3,
        #3, 1], [3, 1, 3], [1, 3, 3]]:
    #for extra_conv_kernel in [[1, 3, 1]]:
            #for multibin_xy_count in [4]:
        yolo_multibin_xy_low = 0.5 / multibin_xy_count
        #multibin_xy_count = 16
        yolo_multibin_xy_high = 1 - 0.5 / multibin_xy_count
        yolo_multibin_xy_count = multibin_xy_count
        #for yolo_obj_kl_distance in [False]:
        yolo_obj_kl_distance = False
    #for yolo_object_scale in [40, 60]:
        #for monitor_train_only in [False, True]:
        #for monitor_train_only in [True]:
        #for yolo_iou_th_to_use_bkg_cls in [0.1]:
        yolo_iou_th_to_use_bkg_cls = 0.1
        #for last_conv_bias in [True]:
        last_conv_bias = True
        #for rotate_max in [5, 10, 15, 20]:
        rotate_with_90 = [[False, True, True, True], [False]]
        rotate_max = [[0, 0, 10, 180], [0]]
        box_data_param_weightss = [[1, 1, 1, 1], [1]]
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, True), (True, False)]:
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, False)]:
        incorporate_at_least_one_box, scale_constrained_by_one_box_area = False, False
        #for yolo_angular_loss_weight in [0.1, 1, 10]:
        #for yolo_angular_loss_weight in [1]:
        yolo_angular_loss_weight = 1
        #for data in ['fridge_clean', 'voc20', 'CARPK']:
        #for data in ['office_v2.1']:
        for data in ['CARPK_select.5.5.nokeep_R905_NMS0.2']:
        #for data in ['CARPK_select.5.5.nokeep']:
        #for data in ['CARPK_select.5.5.nokeep_iter1_iter1_iter1']:
        #for data in ['CARPK_select.5.5.nokeep.tilebb']:
        #for data in ['CARPK_select.5.100000.nokeep']:
        #for data in ['voc20']:
        #for data in ['coco_phone']:
        #for data in ['coco2017']:
            if len(low_shot_label) > 0:
                dataset_ops = [{'op':'low_shot', 'labels': 'dog', 'num_train': 1}]
                dataset_ops[0]['labels'] = low_shot_label
                dataset_ops[0]['labels_idx'] = full_labels.index(low_shot_label)
            else:
                dataset_ops = [{'op':'remove'},
                        {'op':'add',
                         'name': data,
                         'source':'train',
                         'weight': 1},
                        {'op': 'add',
                         'name': 'voc20_removelabel',
                         'source': 'train',
                         'weight': 1},
                        ]
                #dataset_ops = []
            if data.startswith('office'):
                yolo_tree = True
            else:
                yolo_tree = False
            if test_tree_cls_specific_th_by_average:
                assert yolo_tree
        #for yolo_random_scale_min in [0.5, 0.75]:
        #for monitor_train_only in [False]:
        #monitor_train_only = True
        #monitor_train_only = True
    #for monitor_train_only in [True]:
    #for monitor_train_only in [False, True]:
            #for add_reorg, num_extra_convs in [(True, [6]), (True, [7])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(False, [3]), (True, [3])]:
            #for add_reorg, num_extra_convs in [(True, [3])]:
            for add_reorg, num_extra_convs in [(True, [3])]:
        #for add_reorg, num_extra_convs in all_extra_option:
        #for weight_decay in [0.0001, 0.00005]:
        #for weight_decay in [0.0005]:
            #weight_decay = 0.0005
            #weight_decay = 0
            #for net in ['darknet19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
            #for net in ['resnet34']:
            #net = 'darknet19'
            #for net in ['resnet101']:
            #for yolo_coord_scale in [1]:
                yolo_coord_scale = 1
                #for data in ['voc20', 'voc2012', 'coco2017', 'imagenet']:
                #for data in ['coco2017']:
                #for weight_decay in [0.001, 0.005, 0.01, 0.05]:
                weight_decay = 0.0005
                #for net in ['resnet34', 'darknet19_448']:
                #for net in ['darknet19_448', 'resnet34']:
                for net in ['darknet19_448']:
                #for net in ['resnet101']:
                #for net in ['darknet19_448', 'resnet34', 'resnet101']:
                    if len(burn_in) > 0 and data != 'voc20':
                        kwargs_template['stageiter'] = ['60e', '90e', '900e'] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()
                    elif len(burn_in) > 0:
                        kwargs_template['stageiter'] = [5000, 9000, 100000] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()

                    kwargs = copy.deepcopy(kwargs_template)
                    expid = expid_prefix
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if len(num_extra_convs) == 0 and num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                        kwargs['num_extra_convs'] = num_extra_convs
                    elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3 \
                            or len(num_extra_convs) > 1:
                        expid = expid + '_extraconv{}'.format(
                                '_'.join(map(str, num_extra_convs)))
                        kwargs['num_extra_convs'] = num_extra_convs
                    if multibin_wh:
                        expid = expid + '_multibin_wh_{}_{}_{}'.format(multibin_wh_low,
                                multibin_wh_high, multibin_wh_count)
                        kwargs['multibin_wh'] = multibin_wh
                        kwargs['multibin_wh_low'] = multibin_wh_low
                        kwargs['multibin_wh_high'] = multibin_wh_high
                        kwargs['multibin_wh_count'] = multibin_wh_count
                    if not add_reorg:
                        kwargs['add_reorg'] = add_reorg
                    if num_anchor != 5:
                        expid = '{}_numAnchor{}'.format(expid, num_anchor)
                        assert len(anchor_bias) == 2 * num_anchor
                        kwargs['anchor_bias'] = anchor_bias
                    if not yolo_rescore:
                        expid = '{}_{}'.format(expid, 'norescore')
                        kwargs['yolo_rescore'] = yolo_rescore
                    if yolo_obj_ignore_center_around:
                        expid = '{}_{}'.format(expid, 'ignore')
                        kwargs['yolo_obj_ignore_center_around'] = yolo_obj_ignore_center_around
                    if yolo_obj_kl_distance:
                        expid = '{}_{}'.format(expid, 'objkl')
                        kwargs['yolo_obj_kl_distance'] = yolo_obj_kl_distance
                    if yolo_xy_kl_distance:
                        expid = '{}_xykl'.format(expid)
                        kwargs['yolo_xy_kl_distance'] = yolo_xy_kl_distance
                    if yolo_obj_only:
                        expid = '{}_objonly'.format(expid)
                        kwargs['yolo_obj_only'] = yolo_obj_only
                    if yolo_exp_linear_wh:
                        expid = '{}_explinearwh'.format(expid)
                        kwargs['yolo_exp_linear_wh'] = yolo_exp_linear_wh
                    if yolo_obj_nonobj_align_to_iou:
                        expid = '{}_nonobjtoiou'.format(expid)
                        kwargs['yolo_obj_nonobj_align_to_iou'] = yolo_obj_nonobj_align_to_iou
                    if yolo_obj_set1_center_around:
                        expid = '{}_around1'.format(expid)
                        kwargs['yolo_obj_set1_center_around'] = yolo_obj_set1_center_around
                    if yolo_obj_nonobj_nopenaltyifsmallthaniou:
                        expid = '{}_nolosssmallthaniou'.format(expid)
                        kwargs['yolo_obj_nonobj_nopenaltyifsmallthaniou'] = yolo_obj_nonobj_nopenaltyifsmallthaniou
                    if yolo_obj_cap_center_around:
                        expid = '{}_objcaparound'.format(expid)
                        kwargs['yolo_obj_cap_center_around'] = yolo_obj_cap_center_around
                    if weight_decay != 0.0005:
                        expid = '{}_decay{}'.format(expid, weight_decay)
                        kwargs['weight_decay'] = weight_decay
                    if yolo_fixed_target:
                        expid = '{}_fixedtarget'.format(expid)
                        kwargs['yolo_fixed_target'] = yolo_fixed_target
                    if yolo_deconv_to_increase_dim:
                        expid = '{}_deconvincreasedim'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim'] = True
                    if yolo_coord_scale != 1:
                        expid = '{}_coordscale{}'.format(expid,
                                yolo_coord_scale)
                        kwargs['yolo_coord_scale'] = yolo_coord_scale
                    if not yolo_deconv_to_increase_dim_adapt_bias:
                        expid = '{}_nobiaseadapt'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim_adapt_bias'] = \
                            False
                    if yolo_anchor_aligned_images != 12800:
                        expid = '{}_align{}'.format(expid,
                                yolo_anchor_aligned_images)
                        kwargs['yolo_anchor_aligned_images'] = yolo_anchor_aligned_images
                    if yolo_nonobj_extra_power != 0:
                        expid = '{}_nonobjpower{}'.format(expid,
                                yolo_nonobj_extra_power)
                        kwargs['yolo_nonobj_extra_power'] = yolo_nonobj_extra_power
                    if yolo_obj_extra_power != 0:
                        expid = '{}_objpower{}'.format(expid,
                                yolo_obj_extra_power)
                        kwargs['yolo_obj_extra_power'] = yolo_obj_extra_power
                    if yolo_multibin_xy:
                        kwargs['yolo_multibin_xy'] = yolo_multibin_xy
                        kwargs['yolo_multibin_xy_low'] = yolo_multibin_xy_low
                        kwargs['yolo_multibin_xy_high'] = yolo_multibin_xy_high
                        kwargs['yolo_multibin_xy_count'] = yolo_multibin_xy_count
                        expid = '{}_multibinXY{}'.format(expid,
                                yolo_multibin_xy_count)
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'select_top':
                        expid = '{}_selectTop{}'.format(expid,
                                dataset_ops[0]['num_top'])
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'low_shot':
                        low_shot_labels = dataset_ops[0]['labels']
                        low_shot_num_train = dataset_ops[0]['num_train']
                        expid = '{}_lowShot.{}.{}'.format(expid, 
                                low_shot_labels,
                                low_shot_num_train)
                    if len(dataset_ops) > 0:
                        kwargs['dataset_ops'] = dataset_ops
                        expid = '{}_dataop{}'.format(expid,
                                hash(json.dumps(dataset_ops))%10000)
                    if yolo_disable_data_augmentation:
                        expid = '{}_noAugmentation'.format(expid)
                    if yolo_disable_data_augmentation_except_shift:
                        expid = '{}_noAugExpShift'.format(expid)
                    if bn_no_train:
                        expid = '{}_bnNoTrain'.format(expid)
                        kwargs['bn_no_train'] = bn_no_train
                    if yolo_object_scale != 5:
                        expid = '{}_objscale{}'.format(expid, yolo_object_scale)
                        kwargs['yolo_object_scale'] = yolo_object_scale
                    if yolo_noobject_scale != 1:
                        expid = '{}_noobjScale{}'.format(expid,
                                yolo_noobject_scale)
                        kwargs['yolo_noobject_scale'] = yolo_noobject_scale
                    if yolo_class_scale != 1:
                        expid = '{}_clsScale{}'.format(expid,
                                yolo_class_scale)
                        kwargs['yolo_class_scale'] = yolo_class_scale
                    if yolo_avg_replace_max:
                        expid = '{}_avgReplaceMax'.format(expid)
                        kwargs['yolo_avg_replace_max'] = yolo_avg_replace_max
                    if not yolo_sigmoid_xy:
                        expid = '{}_nosigmoidXY'.format(expid)
                        kwargs['yolo_sigmoid_xy'] = yolo_sigmoid_xy
                    if yolo_delta_region3:
                        expid = '{}_deltaRegion3'.format(expid)
                        kwargs['yolo_delta_region3'] = yolo_delta_region3
                    if yolo_background_class:
                        expid = '{}_bkgCls{}'.format(expid,
                                yolo_use_background_class_to_reduce_obj)
                        kwargs['yolo_background_class'] = True
                        kwargs['yolo_use_background_class_to_reduce_obj'] = yolo_use_background_class_to_reduce_obj
                        if yolo_iou_th_to_use_bkg_cls != 1:
                            expid = '{}_iouTh{}'.format(expid,
                                    yolo_iou_th_to_use_bkg_cls)
                            kwargs['yolo_iou_th_to_use_bkg_cls'] = yolo_iou_th_to_use_bkg_cls
                    if res_loss:
                        expid = '{}_resLoss'.format(expid)
                        kwargs['res_loss'] = res_loss
                        kwargs['skip_genprototxt'] = True
                    if yolo_per_class_obj:
                        expid = '{}_perClassObj'.format(expid)
                        kwargs['yolo_per_class_obj'] = yolo_per_class_obj
                    if not last_conv_bias:
                        expid = '{}_noBiasLastConv'.format(expid)
                        kwargs['yolo_last_conv_bias'] = last_conv_bias
                    if yolo_low_shot_regularizer:
                        expid = '{}_lowShotEqualNorm'.format(expid)
                        kwargs['yolo_low_shot_regularizer'] = True
                    if yolo_full_gpu:
                        expid = '{}_fullGpu'.format(expid)
                        kwargs['yolo_full_gpu'] = yolo_full_gpu
                    if burn_in != '':
                        expid = '{}_burnIn{}.{}'.format(expid, burn_in,
                                burn_in_power)
                        kwargs['burn_in'] = burn_in
                        kwargs['burn_in_power'] = burn_in_power
                    if len(rotate_max) != 1 or len(rotate_max[0]) != 1 or \
                            rotate_max[0][0] != 0:
                        kwargs['rotate_max'] = rotate_max
                        expid = '{}_rotate{}'.format(expid, '.'.join([str(y) 
                            for x in rotate_max for y in x]))
                    if len(rotate_with_90) != 1 or \
                            len(rotate_with_90[0]) != 1 or \
                            rotate_with_90[0][0]:
                        expid = '{}_r90.{}'.format(expid, '.'.join(['1' if x else '0' for y in
                                rotate_with_90 for x in y]))
                        kwargs['rotate_with_90'] = rotate_with_90
                    if len(box_data_param_weightss) != 1 or \
                            len(box_data_param_weightss[0]) != 1 or \
                            box_data_param_weightss[0][0] != 1:
                        kwargs['box_data_param_weightss'] = box_data_param_weightss
                        expid = '{}_boxWeight.{}'.format(expid, '.'.join([str(y) for x in
                                box_data_param_weightss for y in x]))
                    if yolo_random_scale_min != 0.25:
                        expid = '{}_randomScaleMin{}'.format(expid, 
                                yolo_random_scale_min)
                        kwargs['yolo_random_scale_min'] = yolo_random_scale_min
                    if yolo_random_scale_max != 2:
                        expid = '{}_randomScaleMax{}'.format(expid,
                                yolo_random_scale_max)
                        kwargs['yolo_random_scale_max'] = yolo_random_scale_max
                    if scale_relative_input:
                        expid = '{}_RelativeScale2'.format(expid)
                        kwargs['scale_relative_input'] = scale_relative_input
                    if nms_type != 'Standard' and nms_type != '':
                        if nms_type == 'LinearSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.LinearSoft 
                        if nms_type == 'GaussianSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.GaussianSoft
                        if gaussian_nms_sigma != 0.5:
                            kwargs['gaussian_nms_sigma'] = 0.5
                    if incorporate_at_least_one_box:
                        expid = '{}_atLeastOneBB'.format(expid)
                        kwargs['incorporate_at_least_one_box'] = incorporate_at_least_one_box
                    if scale_constrained_by_one_box_area:
                        expid = '{}_scaleConstrainedByOne'.format(expid)
                        kwargs['scale_constrained_by_one_box_area'] = scale_constrained_by_one_box_area
                        if scale_constrained_by_one_box_area_min != 0.001:
                            expid = '{}_Min{}'.format(expid, scale_constrained_by_one_box_area_min)
                            kwargs['scale_constrained_by_one_box_area_min'] = scale_constrained_by_one_box_area_min
                    if yolo_tree:
                        expid = '{}_tree'.format(expid)
                        kwargs['yolo_tree'] = yolo_tree
                        if test_tree_cls_specific_th_by_average is not None:
                            kwargs['test_tree_cls_specific_th_by_average'] = test_tree_cls_specific_th_by_average
                    if len(init_from) > 0:
                        assert net == init_from['net']
                        c = CaffeWrapper(data=init_from['data'], 
                                net=init_from['net'],
                                expid=init_from['expid'])
                        kwargs['basemodel'] = c.best_model().model_param
                        expid = '{}_init{}'.format(expid,
                                hash(kwargs['basemodel']) % 10000)
                    if yolo_angular_loss:
                        expid = '{}_AngularRegulizer'.format(expid)
                        kwargs['yolo_angular_loss'] = True
                        if yolo_angular_loss_weight != 1:
                            expid = '{}Weight{}'.format(expid,
                                    yolo_angular_loss_weight)
                            kwargs['yolo_angular_loss_weight'] = yolo_angular_loss_weight
                    if len(no_bias) > 0:
                        expid = '{}_noBias{}'.format(expid, no_bias)
                        kwargs['no_bias'] = no_bias
                    if net_input_size_min != 416:
                        expid = '{}_InMin{}'.format(expid, net_input_size_min)
                        kwargs['net_input_size_min'] = net_input_size_min
                    if net_input_size_max != 416:
                        expid = '{}_InMax{}'.format(expid, net_input_size_max)
                        kwargs['net_input_size_max'] = net_input_size_max
                    if any(k != 3 for k in extra_conv_kernel):
                        expid = '{}_extraConvKernel.{}'.format(expid, '.'.join(
                            map(str, extra_conv_kernel)))
                        kwargs['extra_conv_kernel'] = extra_conv_kernel
                    if any(c != 1024 for c in extra_conv_channels):
                        expid = '{}_extraChannels.{}'.format(expid,
                                '.'.join(map(str, extra_conv_channels)))
                        kwargs['extra_conv_channels'] = extra_conv_channels
                    if len(last_fixed_param) > 0:
                        expid = '{}_FixParam.{}'.format(expid,
                                last_fixed_param.replace('/', '.'))
                        kwargs['last_fixed_param'] = last_fixed_param
                    if data.startswith('office_v2.1'):
                       assert 'taxonomy_folder' not in kwargs
                       kwargs['taxonomy_folder'] = \
                            './aux_data/taxonomy10k/office/{}'.format(data)
                    if residual_loss:
                        expid = '{}_resLoss{}'.format(expid, 
                                '.'.join(map(lambda x: x.replace('/', '_'),
                                    residual_loss_froms)))
                        kwargs['residual_loss'] = residual_loss
                        kwargs['residual_loss_froms'] = residual_loss_froms
                    if yolo_disable_no_penalize_if_iou_large:
                        kwargs['yolo_disable_no_penalize_if_iou_large'] = True
                        expid = '{}_disableNoPenIfIouLarge'.format(expid)
                    if cutout_prob > 0:
                        kwargs['cutout_prob'] = cutout_prob
                        expid = '{}_cutoutProb{}'.format(expid, cutout_prob)
                    if len(multi_feat_anchor) > 0:
                        kwargs['multi_feat_anchor'] = multi_feat_anchor
                        expid = '{}_multiFeatAnchor{}.{}.{}'.format(expid,
                                len(multi_feat_anchor),
                                '.'.join(map(str, [m['loss_weight_multiplier'] for m in
                                    multi_feat_anchor])),
                                hash(json.dumps(multi_feat_anchor)))
                    #if max_iters != '128e' and max_iters != 10000:
                        #expid = '{}_maxIter.{}'.format(expid, max_iters)
                    if yolo_softmax_norm_by_valid:
                        expid = '{}_softmaxByValid'.format(expid)
                        kwargs['yolo_softmax_norm_by_valid'] = yolo_softmax_norm_by_valid
                    #if yolo_softmax_extra_weight != 1:
                        #expid = '{}_softmaxWeight{}'.format(expid,
                                #yolo_softmax_extra_weight)
                        #kwargs['yolo_softmax_extra_weight'] = yolo_softmax_extra_weight
                    if ignore_negative_first_batch:
                        expid = '{}_ignoreNegativeFirst'.format(expid)
                        kwargs['ignore_negative_first_batch'] = ignore_negative_first_batch
                        if yolo_not_ignore_negative_seen_images > 0:
                            expid = '{}_notIgnore{}'.format(expid,
                                    yolo_not_ignore_negative_seen_images)
                            kwargs['yolo_not_ignore_negative_seen_images'] = yolo_not_ignore_negative_seen_images
                        if yolo_force_negative_with_partial_overlap:
                            expid = '{}_ForcePartial'.format(expid)
                            kwargs['yolo_force_negative_with_partial_overlap'] = True
                    if yolo_nms != 0.45:
                        kwargs['yolo_nms'] = yolo_nms
                    if first_batch_objectiveness_enhancement:
                        expid = '{}_FirstObjEnhance{}'.format(expid,
                                first_batch_objectiveness_enhancement_weight)
                        kwargs['first_batch_objectiveness_enhancement'] = True
                        kwargs['first_batch_objectiveness_enhancement_weight'] = first_batch_objectiveness_enhancement_weight
                    expid = expid + suffix
                    kwargs['monitor_train_only'] = monitor_train_only
                    kwargs['expid'] = expid
                    kwargs['net'] = net
                    kwargs['data'] = data
                    all_task.append(kwargs)

    all_resource = get_all_resources()
    logging.info(pformat(all_resource))
    logging.info(pformat(all_task))
    logging.info('#resource: {}'.format(len(all_resource)))
    logging.info('#task: {}'.format(len(all_task)))
    debug = True
    #all_task[0]['force_predict'] = True
    debug = False
    #return
    #tsv_file = './data/office_v2.1_with_bb/test.tsv'
    #all_task[0]['force_predict'] = True
    #task = all_task[0]
    #task['expid'] = '{}_bb_nobb'.format(task['expid'])
    ##task['expid'] = '{}_bb_only'.format(task['expid'])
    #task['class_specific_nms'] = False
    #task['yolo_test_thresh'] = 0.5
    #c = CaffeWrapper(**task)
    #c.demo(None)
    #rows = tsv_reader(tsv_file)
    #for row in rows:
        #continue
    #import ipdb;ipdb.set_trace()
    #c.demo('./data/office100_v1_with_bb/train.tsv')
    #c.demo(tsv_file)
    #c.demo('/raid/jianfw/data/office100_crawl/TermList.instagram.pinterest.scrapping.image.tsv')
    #c.demo('/raid/jianfw/work/yuxiao_crop/ring/')
    #c.demo('tmp.png')
    #sall_task[0]['force_predict'] = True
    return
    def batch_run():
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
        #if not monitor_train_only:
            #for t in all_task:
                #t['monitor_train_only'] = True
            #for i, r in enumerate(all_resource):
                #all_resource[i] = (r[0], [-1] * 4)
            #b = BatchProcess(all_resource, all_task, task_processor)
            ##b._availability_check = False
            #b.run()
    if debug:
        idx = -1
        task = all_task[idx]
        task['effective_batch_size'] = 8
        #task['use_pretrained'] = False
        #task['max_iters'] = 1
        #task['expid'] = '{}_debug'.format(expid)
        #all_task[idx]['datas'] = ['voc20', 'crawl_office_v1']
        #task['force_train'] = True
        task['debug_train'] = True
        #all_task[idx]['debug_detect'] = True
        #all_task[idx]['force_predict'] = True
        #task_processor(({}, [0]), all_task[idx])
        #task['force_evaluate'] = True
        task_processor(({}, [0]), task)
        #task_processor(all_resource[-1], task)
        #task_processor((vig[1], [0]), task)
        #import ipdb;ipdb.set_trace()
    else:
        batch_run()
        pass

def yolo_demo():
    data = 'Tax1300SGV1_1'
    net = 'darknet19_448'
    expid = 'B_noreorg_extraconv2_tree_init3491_IndexLossWeight0_bb_nobb'

    data = 'TaxPerson_V1_2'
    net = 'darknet19'
    expid = 'person_bb_only'

    test_tsv = './data/voc20/test.tsv'
    test_tsv = '/mnt/jianfw_desk/photo_0035.jpg'

    c = CaffeWrapper(data, net, load_parameter=True, expid=expid)
    c.demo(test_tsv)

def update_imagenet2012_param(param):
    param['net'] = 'SEBNInception'
    #param['net'] = 'resnet10'
    #param['net'] = 'resnet101'
    param['effective_batch_size'] = 256
    param['base_lr'] = 0.1
    param['lr_policy'] = 'step'
    if param['expid_prefix'] == 'lei' and \
            param['net'] == 'resnet101':
        param['max_iters'] = 650000
    else:
        param['max_iters'] = 450000
    param['predict_style'] = 'tsvdatalayer'
    param['snapshot'] = 5000
    param['stepsize'] = 100000
    param['use_pretrained'] = False
    param['weight_decay'] = 5e-5
    param['test_input_sizes'] = [320, 416, 640]

    param['use_tsvbox_for_cls'] = True
    if not param['use_tsvbox_for_cls']:
        param['crop_type'] = caffe.params.TsvData.InceptionStyle
        param['inception_crop_kl'] = './data/imagenet2012/kl.txt'
    param['use_pretrained'] = False
    if param['expid_prefix'] == 'lei':
        param['skip_genprototxt'] = True
    param['rotate_max'] = [[10]]

def update_param_by_data(data, param):
    if data == 'imagenet2012':
        update_imagenet2012_param(param)

def yolo_master(**param):
    data = 'CARPK'
    #data = 'CARPK_select.5.100000.nokeep'
    data = 'CARPK_select.5.5.nokeep'
    data = 'voc20'
    #data = 'Tax700V2_1'
    #data = 'Tax700V2'
    #data = 'imagenet200'
    #data = 'CARPK_select.5.5.nokeep'
    #data = 'CARPK_select.5.5.nokeep_R905_NMS0.2'
    #data = 'Tax700V3_1'
    #data = 'WIDER_FACE1024'
    #data = 'voc20'
    #data = 'CocoBottle1024Merge'
    #data = 'CARPK_select.10000.5.nokeep'
    #data = 'CARPK_select.500.5.nokeep'
    #data = 'CocoBottle1024Merge_select.10000.5.nokeep'
    #data = 'CocoBottle1024Merge_selectbylabel.10.1000'
    #data = 'CocoBottle1024Merge_selectbylabel.10.1000'
    #data = 'CocoBottle1024Drink_select.10.5.nokeep'
    #data = 'CocoBottle1024DrinkY'
    #data = 'CocoBottle1024DrinkY_select.10.5.nokeep'
    data = 'imagenet200'
    data = 'Tax1300SGV1_1'
    #data = 'vot_ball'
    #data = 'Tax700V3_1_debug'
    #data = 'icdar_e2e_2015_focused'
    data = 'TaxPerson'
    data = 'voc0712'
    #data = 'voc20'
    #data = 'TaxPerson_V1_2_with_bb_S'
    #data = 'TaxPerson_V1_2_S_M1_C_with_bb'
    #data = 'TaxPerson_V1_2_S_C_with_bb'
    #data = 'TaxVocPerson_V1_1'
    data = 'imagenet2012'
    #data = 'cifar10'

    data_with_bb = '{}_with_bb'.format(data)
    data_no_bb = '{}_no_bb'.format(data)
    
    param['net'] = 'darknet19_448'
    if data == 'imagenet2012':
        detmodel = 'classification'
    else:
        detmodel = 'yolo'

    if detmodel == 'yolo':
        #expid_prefix = 'A'
        param['expid_prefix'] = 'B' # B means softmax normalized by valid and gpu
        #expid_prefix = 'debug' # B means softmax normalized by valid and gpu
        if data == 'TaxVocPerson_V1_1':
            param['expid_prefix'] = 'person_bb_only' # B means softmax normalized by valid and gpu
    else:
        param['expid_prefix'] = 'A'
        #param['expid_prefix'] = 'lei'
        #param['expid_prefix'] = 'Official'

    skip_training = False
    #skip_training = True
    
    param['rotate_max'] = [[0]]
    tsv_box_max_samples = [[50]]
    tsv_box_max_samples = [[1]]
    box_data_param_weightss = [[1]]
    ignore_negative_first_batch = False
    yolo_force_negative_with_partial_overlap = False
    #yolo_force_negative_with_partial_overlap = True
    yolo_index_threshold_loss_extra_weight = 1
    if 'test_data' not in param:
        param['test_data'] = data
    #test_data = 'MSLogoClean'
    #test_data = 'brand1048Clean'
    #test_data = 'TaxPerson_V1_2_with_bb_S_M1'
    #test_data = 'voc20_person'
    if 'test_split' not in param:
        param['test_split'] = 'test'
    #param['test_split'] = test_split 
    init_from = {}
    #init_from = {'data': 'imagenet200', 
            #'net': 'darknet19_448', 
            #'expid': 'B_noreorg_extraconv2'}
    dataset_ops = []
    param['max_iters'] = '128e'
    suffix = ''
    num_extra_convs = [1]
    num_extra_convs = [3]
    rotate_with_90 = [[False]]
    add_reorg = False
    add_reorg = True
    ovthresh = [0,0.3,0.5]
    first_batch_objectiveness_enhancement = False
    first_batch_objectiveness_enhancement_weight = 1
    param['yolo_random_scale_min'] = 0.25
    param['yolo_random_scale_max'] = 2

    if data == 'TaxVocPerson_V1_1':
        param['net'] = 'darknet19'

    param['test_input_sizes'] = [320, 416, 640]
    #param['test_input_sizes'] = [416]
    param['test_on_train'] = False
    param['detmodel'] = detmodel
    
    # effective batch size
    if data.startswith('Tax'):
        effective_batch_size = 64 * 2

    if param['test_data'] == data_no_bb:
        ovthresh = [-1]

    update_param_by_data(data, param)
    
    if skip_training:
        param['skip_genprototxt'] = True
        param['load_parameter'] = True

    #param['skip_genprototxt'] = True
    if data.startswith('Tax') and False:
        with_taxonomy_folder = False
        add_reorg = False
        phase = 0
        #phase = 1
        #phase = -1
        #param['restore_snapshot_iter'] = 600

        if data == 'Tax700V3_1':
            init_from = {'data': 'Tax700V2', 
                    'net': param['net'], 
                    'expid': 'B_noreorg_extraconv2_dataop7098_tree_initFrom1416_maxIter.30e_IndexLossWeight0_bb_nobb'}
        elif data == 'Tax1300SGV1_1':
            init_from = {'data': 'Tax700V3_1',
                    'net': 'darknet19_448',
                    'expid': 'B_noreorg_extraconv2_tree_init5494_IndexLossWeight0_bb_nobb'}
        elif data == 'TaxPerson_V1_3':
            param['basemodel'] = 'output/Tax4k_V1_2_darknet19_no_visual_genome_bb_nobb/snapshot/model_iter_660000.caffemodel'

        if phase == 0:
            dataset_ops = [{'op':'remove'},
                    {'op':'add',
                     'name': data_with_bb,
                     'source':'train',
                     'weight': 1},
                    ]
            init_from = {}
            param['max_iters'] = '128e'
            suffix = '_bb_only'
        elif phase == 1:
            dataset_ops = [{'op':'remove'},
                    {'op':'add',
                     'name': data_with_bb,
                     'source':'train',
                     'weight': 3},
                    {'op': 'add',
                     'name': data_no_bb,
                     'source': 'train',
                     'weight': 1},
                    ]
            init_from = {'data': data, 
                    'net': param['net'], 
                    'expid': 'B_noreorg_extraconv2_bb_only'}
            init_from = {}
            #max_iters = '30e'
            suffix = '_bb_nobb_debug'
            yolo_index_threshold_loss_extra_weight = 0
        elif phase == -1:
            dataset_ops = []
            #init_from = {}
            param['max_iters'] = '128e'
            param['tree_max_iters2'] = '30e'
            param['taxonomy_folder'] = \
                './aux_data/taxonomy10k/tax700/Tax700V3'
            yolo_index_threshold_loss_extra_weight = 0
        num_extra_convs = [3]
        if len(num_extra_convs) ==1 and num_extra_convs[0] == 3:
            param['extra_conv_kernel'] = [1, 3, 1]
            param['extra_conv_groups'] = [0, 4, 0]
            init_from = {}
            param['basemodel'] = 'output/iris_basemodel/darknet19_imgnetyolofc3nog4.caffemodel'
        param['test_data'] = data_with_bb
        param['test_data'] = data_no_bb
    elif data.startswith('vot'):
        param['max_iters'] = 0
    elif data.startswith('WIDER_FACE'):
        #tsv_box_max_samples = [[50, 50, 50, 50]]
        #rotate_with_90 = [[False, True, True, True]]
        #rotate_max = [[0, 0, 10, 180]]
        #box_data_param_weightss = [[1, 1, 1, 1]]
        param['yolo_random_scale_min'] = 2
        param['yolo_random_scale_max'] = 8
        param['test_input_sizes'] = [416 * 5]
    elif data.startswith('CocoBottle1024'):
        param['yolo_random_scale_min'] = 2
        param['yolo_random_scale_max'] = 4
        param['test_input_sizes'] = [416 * 3]
        param['max_iters'] = 10000
        dataset_ops = [{'op':'remove'},
                       {'op':'add',
                             'name': data,
                             'source':'train',
                             'weight': 3},
                       {'op': 'add',
                             'name': 'voc20_removelabel',
                             'source': 'train',
                             'weight': 1},
                       ]
        tsv_box_max_samples = [[50], [1]]
        if 'Merge' in data:
            param['test_data'] = 'CocoBottle1024Merge'
        elif 'Drink' in data:
            param['test_data'] = 'CocoBottle1024DrinkYW2'
        else:
            assert False
        #test_data = data
        param['test_on_train'] = False
    elif data.startswith('CARPK'):
        sub = ''
        sub = 'with_background'
        dataset_ops = []
        init_from = {}
        yolo_index_threshold_loss_extra_weight = 1
        suffix = ''
        num_extra_convs = [3]
        #max_iters = 10000
        param['max_iters'] = 10000
        param['test_data'] = 'CARPK'
        param['test_data'] = data

        #ignore_negative_first_batch = True
        if sub == '':
            rotate_with_90 = [[False, True, True, True]]
            param['rotate_max'] = [[0, 0, 10, 180]]
            #box_data_param_weightss = [[1, 1, 1, 1]]
            tsv_box_max_samples = [[50, 50, 50, 50]]
            #rotate_max = [[0, 0, 10, 180], [0]]
            #rotate_with_90 = [[False, True, True, True], [False]]
            box_data_param_weightss = [[1, 1, 1, 1]]
        elif sub == 'with_background':
            car_weight = 1
            dataset_ops = [{'op':'remove'},
                           {'op':'add',
                                 'name': data,
                                 'source':'train',
                                 'weight': car_weight},
                           {'op': 'add',
                                 'name': 'voc20_removelabel',
                                 'source': 'train',
                                 'weight': 1},
                           ]
            if car_weight != 1:
                paam['expid_prefix'] = '{}CarWeight{}'.format(expid_prefix, car_weight)
            rotate_with_90 = [[False, True, True, True], [False]]
            param['rotate_max'] = [[0, 0, 10, 180], [0]]
            box_data_param_weightss = [[1, 1, 1, 1], [1]]
            #tsv_box_max_samples = [[50, 50, 50, 50], [1]]
        if data == 'CARPK_select.5.5.nokeep_R905_NMS0.2':
            init_from = {'data': 'CARPK_select.5.5.nokeep_R905_NMS0.2',
                    'net': 'darknet19_448', 
                    'expid': 'R90'}
    elif data == 'imagenet2012':
        pass


    if data.startswith('office'):
        param['yolo_tree'] = True
    elif data.startswith('Tax'):
        param['yolo_tree'] = True
    else:
        param['yolo_tree'] = False
    
    if data == 'TaxVocPerson_V1_1' and param['net'] == 'darknet19':
        add_reorg = True
        num_extra_convs = [3]

    # some quick experiment
    #param['yolo_tree'] = True
    #param['softmax_tree_prediction_threshold'] = 0.001

    monitor_train_only = True
    #monitor_train_only = False
    #task_type = 'debug_batch_run'
    task_type = 'print_info'
    task_type = 'speed'
    task_type = 'none'
    task_type = 'batch_run'
    task_type = 'demo'
    task_type = 'debug_batch_run'
    task_type = 'debug_train'
    #task_type = 'debug_predict'
    #task_type = 'debug_evaluate'
    #task_type = 'debug'
    task_type = ''
    task_type = 'batch_run'
    if len(sys.argv) == 2:
        logging.info('change to batch run')
        task_type = 'batch_run'
    task_param = yolo_master_task(data=data,
            task_type=task_type,
            init_from=init_from,
            dataset_ops=dataset_ops,
            suffix=suffix,
            monitor_train_only=monitor_train_only,
            yolo_index_threshold_loss_extra_weight=yolo_index_threshold_loss_extra_weight,
            num_extra_convs=num_extra_convs,
            rotate_with_90=rotate_with_90,
            box_data_param_weightss=box_data_param_weightss,
            ignore_negative_first_batch=ignore_negative_first_batch,
            yolo_force_negative_with_partial_overlap=yolo_force_negative_with_partial_overlap,
            tsv_box_max_samples=tsv_box_max_samples,
            add_reorg=add_reorg,
            ovthresh=ovthresh,
            first_batch_objectiveness_enhancement=first_batch_objectiveness_enhancement,
            first_batch_objectiveness_enhancement_weight=first_batch_objectiveness_enhancement_weight,
            **param
            )
    logging.info(pformat(task_param))

def yolo_master_task(**param):
    data = param['data']
    test_data = param['test_data']
    rotate_with_90 = param['rotate_with_90']
    rotate_max = param['rotate_max']
    box_data_param_weightss = param['box_data_param_weightss']
    expid_prefix = param['expid_prefix']
    ignore_negative_first_batch = param['ignore_negative_first_batch']
    yolo_force_negative_with_partial_overlap = param['yolo_force_negative_with_partial_overlap']
    yolo_index_threshold_loss_extra_weight = param['yolo_index_threshold_loss_extra_weight']
    all_task = []
    effective_batch_size = param.get('effective_batch_size', 64)
    tsv_box_max_samples = param['tsv_box_max_samples']
    add_reorg = param['add_reorg']
    first_batch_objectiveness_enhancement = param['first_batch_objectiveness_enhancement']
    test_on_train = param['test_on_train']
    ovthresh = param['ovthresh']
    def gen_anchor_bias(num_anchor):
        if num_anchor == 0:
            return None
        if num_anchor == 2:
            return [4, 8, 8, 4]
        result = []
        n = int(np.sqrt(num_anchor))
        assert n * n == num_anchor
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 9
    anchor_bias = gen_anchor_bias(num_anchor)
    anchor_bias = None
    multi_feat_anchor = [
            #{   
                #'feature': 'dark4c/leaky', 
                #'extra_conv_channels': [512, 1024],
                #'extra_conv_kernels': [3, 3],
                #'loss_weight_multiplier': 1,
                #'anchor_bias': [1, 1, 1, 2, 2, 1, 2, 2], 
            #},
            {
                'feature': 'dark5e/leaky', 
                'extra_conv_channels': [1024],
                'extra_conv_kernels': [3],
                #'anchor_bias': [4, 4, 2, 4, 4, 2], 
                'loss_weight_multiplier': 1,
                'anchor_bias': [2, 2, 1, 2, 2, 1], 
            }, 
            {   
                'feature': 'dark6e/leaky', 
                'extra_conv_channels': [1024, 1024, 1024],
                'extra_conv_kernels': [3, 3, 3],
                'loss_weight_multiplier': 1,
                #'anchor_bias': [2, 2, 4, 4, 4, 8, 8, 4, 8, 8],
                #'anchor_bias': [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
                'anchor_bias': [3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
            }
            ]
    #multi_feat_anchor = [
            #{   
                #'feature': 'dark6e/leaky', 
                #'extra_conv_channels': [1024, 1024, 1024],
                #'extra_conv_kernels': [3, 3, 3],
                #'loss_weight_multiplier': 1,
                ##'stride': 32,
                ##'anchor_bias': [6.63,11.38,9.42,5.11,16.62,10.52]
                #'anchor_bias': [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
                ##'anchor_bias': [4, 4, 2, 4, 4, 2]
            #},
            #]
    multi_feat_anchor = []
    num_anchor = 5
    multibin_wh = False
    multibin_wh_low = 0;
    multibin_wh_high = 13
    all_multibin_wh_count = [16]
    #all_multibin_wh_count = [16, 32, 48, 64];
    yolo_rescore = True
    yolo_xy_kl_distance = False
    yolo_obj_only = False
    yolo_exp_linear_wh = False
    yolo_obj_nonobj_align_to_iou = False
    yolo_obj_ignore_center_around = False
    yolo_obj_kl_distance = False
    yolo_obj_set1_center_around = False
    yolo_blame = 'xy.wh.obj.cls.nonobj'
    yolo_blame = 'xy.wh.obj.nonobj.cls'
    yolo_blame = ''
    yolo_deconv_to_increase_dim = False
    yolo_deconv_to_increase_dim_adapt_bias = True
    yolo_anchor_aligned_images = 1280000000
    yolo_anchor_aligned_images = 12800
    #yolo_anchor_aligned_images = 0
    yolo_nonobj_extra_power = 0
    yolo_obj_extra_power = 0
    yolo_disable_data_augmentation = False
    yolo_disable_data_augmentation_except_shift = False
    #yolo_anchor_aligned_images = 0
    bn_no_train = False
    yolo_coords_only = False
    if yolo_coords_only:
        yolo_object_scale = 0
        yolo_noobject_scale = 0
        yolo_class_scale = 0
    else:
        yolo_object_scale = 5
        yolo_noobject_scale = 1
        yolo_class_scale = 1
    yolo_avg_replace_max = False
    yolo_per_class_obj = False

    max_iters = param['max_iters']
    if data == 'voc20':
        max_iters = 10000
    elif data == 'fridge_clean':
        max_iters = 10000
    elif data.startswith('CARPK'):
        max_iters = 10000

    burn_in = '5e'
    burn_in = ''
    burn_in_power = 1
    detmodel = param.get('detmodel', 'yolo')
    
    if detmodel == 'yolo':
        yolo_full_gpu = True
    #yolo_full_gpu = False
    test_tree_cls_specific_th_by_average = 1.2
    test_tree_cls_specific_th_by_average = None
    yolo_angular_loss = True
    yolo_angular_loss = False
    no_bias = 'conf'
    no_bias = ''
    num_bn_fix = 0
    test_input_sizes = param['test_input_sizes']
    #test_input_sizes = [320]
    if data.startswith('CARPK'):
        test_input_sizes = [416 * 3]
        #test_input_sizes=[288, 416, 480, 608],
        #test_input_sizes=[2080],
        #test_input_sizes=[416, 544, 608, 992, 1024],
        #test_input_sizes=[992],
        #test_input_sizes=[416 * 3],
    kwargs = dict(
            detmodel=detmodel,
            max_iters=max_iters,
            #max_iters='80e',
            #max_iters='256e',
            #max_iters=max_iters,
            #max_iters=10000,
            #yolo_blame=yolo_blame,
            #expid=expid,
            #yolo_jitter=0,
            #yolo_hue=0,
            #yolo_test_fix_xy = True,
            #yolo_test_fix_wh = True,
            #yolo_extract_target_prediction = True,
            #yolo_max_truth=300,
            #yolo_exposure=1,
            #test_on_train = True,
            #yolo_saturation=1,
            #expid='A_multibin_wh_0_13_16_no2-wh',
            #expid='baseline_2',
            #snapshot=1000,
            #snapshot=0,
            #target_synset_tree='./aux_data/yolo/9k.tree',
            #target_synset_tree='./data/{}/tree.txt'.format(data),
            #dataset_ops=dataset_ops,
            #effective_batch_size=1,
            #num_anchor=3,
            #num_anchor=num_anchor,
            #force_train=True,
            #force_evaluate=True,
            #debug_detect=True,
            #force_predict=True,
            #extract_features='angular_loss.softmax_loss.o_obj_loss.xy_loss.wh_loss.o_noobj_loss',
            #data_dependent_init=True,
            #restore_snapshot_iter=-1,
            #display=0,
            #region_debug_info=10,
            #display=100,
            #stagelr=stagelr,
            #anchor_bias=anchor_bias,
            #stageiter=[1000, 1000000],
            #stageiter=[100,5000,9000,10000000],
            #stageiter = (np.asarray([5000, 9000, 1000000]) / batch_size_factor).tolist(),
            #stagelr = (np.asarray([0.001,0.0001,0.00001]) * 100).tolist(),
            #stagelr=[0.0001,0.001,0.0001,0.0001],
            #burn_in=100,
            #class_specific_nms=False,
            #basemodel='./output/imagenet_darknet19_448_A/snapshot/model_iter_570640.caffemodel',
            #effective_batch_size=effective_batch_size,
            #solver_debug_info=True,
            ovthresh=ovthresh)


    keys = ['lr_policy', 'snapshot',
            'base_lr', 'stepsize',
            'crop_type', 'inception_crop_kl',
            'use_pretrained']

    for key in keys:
        if key in param:
            kwargs[key] = param[key]
    
    
    if detmodel == 'yolo':
        kwargs['yolo_test_maintain_ratio'] = True
    if 'tree_max_iters2' in param:
        kwargs['tree_max_iters2'] = param['tree_max_iters2']

    if 'restore_snapshot_iter' in param:
        kwargs['restore_snapshot_iter'] = param['restore_snapshot_iter']

    #for k in param:
        #kwargs_template[k] = param[k]

    if test_on_train:
        kwargs['test_on_train'] = True

    if effective_batch_size != 64:
        kwargs['effective_batch_size'] = effective_batch_size

    if yolo_disable_data_augmentation:
        kwargs['yolo_jitter'] = 0
        kwargs['yolo_hue'] = 0
        kwargs['yolo_exposure'] = 1
        kwargs['yolo_saturation'] = 1
        kwargs['yolo_random_scale_min'] = 1
        kwargs['yolo_random_scale_max'] = 1
        kwargs['yolo_fix_offset'] = True
        kwargs['yolo_mirror'] = False
    elif yolo_disable_data_augmentation_except_shift:
        kwargs['yolo_jitter'] = 0
        kwargs['yolo_hue'] = 0
        kwargs['yolo_exposure'] = 1
        kwargs['yolo_saturation'] = 1
        kwargs['yolo_random_scale_min'] = 1
        kwargs['yolo_random_scale_max'] = 1

    continue_less_data_augmentation = False
    if continue_less_data_augmentation:
        kwargs['yolo_jitter'] = 0
        kwargs['yolo_hue'] = 0
        kwargs['yolo_exposure'] = 1
        kwargs['yolo_saturation'] = 1
        kwargs['yolo_random_scale_min'] = 1
        kwargs['yolo_random_scale_max'] = 1

    
    yolo_random_scale_min = param['yolo_random_scale_min']
    yolo_random_scale_max = param['yolo_random_scale_max']
    if data.startswith('CARPK'):
        yolo_random_scale_min = 2
        yolo_random_scale_max = 4

    
    #yolo_random_scale_min = 2
    #yolo_random_scale_max = 4
    multibin_wh_count = all_multibin_wh_count[0]
    yolo_fixed_target = False
    yolo_obj_nonobj_nopenaltyifsmallthaniou = False
    yolo_obj_cap_center_around = False
    yolo_multibin_xy = False
    yolo_sigmoid_xy = True
    yolo_delta_region3 = False
    yolo_background_class = False
    yolo_use_background_class_to_reduce_obj = 0.4
    multibin_xy_count = 4
    #res_loss = True
    res_loss = False
    yolo_use_background_class_to_reduce_obj = 1
    monitor_train_only = param['monitor_train_only']
    #for yolo_use_background_class_to_reduce_obj in [1, 0.8, 0.6, 0.4, 0.2]:
    dataset = TSVDataset(data)
    yolo_low_shot_regularizer = False
    #full_labels = dataset.load_labelmap()
    full_labels = []
    #for low_shot_label in full_labels:
    scale_relative_input = True
    scale_relative_input = False
    nms_type = 'LinearSoft'
    nms_type = 'GaussianSoft'
    nms_type = ''
    gaussian_nms_sigma = 0.5
    scale_constrained_by_one_box_area_min = 0.001
    last_fixed_param = 'dark5e/leaky'
    last_fixed_param = ''
    #for low_shot_label in ['']:
    #for low_shot_label in [full_labels[0]]:
    full_labels.insert(0, '')
    #for low_shot_label in full_labels[:5]:
    low_shot_label = ''
    residual_loss = True
    residual_loss = False
    residual_loss_froms = ['extra_conv19/leaky', 'extra_conv20/leaky']
    #for low_shot_label in ['']:
    yolo_disable_no_penalize_if_iou_large = False
    cutout_prob = -1
    #expid_prefix = 'debug3'
    #net_input_size_min = 416 * 2
    #net_input_size_max = 416 * 2
    net_input_size_min = 416
    net_input_size_max = 416
    #yolo_softmax_extra_weight = 0.2
    yolo_not_ignore_negative_seen_images = 0
    yolo_xywh_norm_by_weight_sum = False
    yolo_softmax_extra_weight = 0.2
    yolo_softmax_extra_weight = 1
    #for extra_conv_channels in [[1024, 512, 512], [512, 512, 1024], [512, 1024,
        #512]]:
    extra_conv_channels = [1024, 1024, 1024]
    if len(low_shot_label) > 0:
        dataset_ops = [{'op':'low_shot', 'labels': 'dog', 'num_train': 1}]
        dataset_ops[0]['labels'] = low_shot_label
        dataset_ops[0]['labels_idx'] = full_labels.index(low_shot_label)
    else:
        dataset_ops = param['dataset_ops']
        #for multibin_xy_count in [4]:
    yolo_multibin_xy_low = 0.5 / multibin_xy_count
    #multibin_xy_count = 16
    yolo_multibin_xy_high = 1 - 0.5 / multibin_xy_count
    yolo_multibin_xy_count = multibin_xy_count
    #for yolo_obj_kl_distance in [False]:
    yolo_obj_kl_distance = False
    #for yolo_iou_th_to_use_bkg_cls in [0.1]:
    yolo_iou_th_to_use_bkg_cls = 0.1
    #for last_conv_bias in [True]:
    last_conv_bias = True
    #for rotate_max in [5, 10, 15, 20]:
    #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, True), (True, False)]:
    #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, False)]:
    incorporate_at_least_one_box, scale_constrained_by_one_box_area = False, False
    #for yolo_angular_loss_weight in [0.1, 1, 10]:
    #for yolo_angular_loss_weight in [1]:
    yolo_angular_loss_weight = 1
    #for data in ['fridge_clean', 'voc20', 'CARPK']:
    #for data in ['office_v2.1']:
    #for data in ['CARPK_select.5.5.nokeep']:
    #data = 'Tax700V2'
    #for data in ['voc20', 'voc2012', 'coco2017', 'imagenet']:
    #for data in ['coco_phone']:
    #for data in ['coco2017']:

    yolo_tree = param.get('yolo_tree', False)
    if test_tree_cls_specific_th_by_average:
        assert yolo_tree
    
    if detmodel == 'yolo':
        if data == 'Tax700V1':
            yolo_softmax_norm_by_valid = False
        else:
            yolo_softmax_norm_by_valid = True
            #yolo_softmax_norm_by_valid = False

    if data.startswith('Tax700V1') or data.startswith('Tax700V2'):
        if data.startswith('Tax700V1'):
            taxonomy_folder = './aux_data/taxonomy10k/tax700/tax700_v1'
        elif data.startswith('Tax700V2'):
            taxonomy_folder = './aux_data/taxonomy10k/tax700/tax700_v2'
        else:
            assert False

        if data == 'Tax700V2':
            build_taxonomy_impl(taxonomy_folder,
                    data=data,
                    datas=['coco2017', 
                        'voc0712', 
                        'imagenet3k_448', 
                        'imagenet22k_448', 
                        'crawl_office_v1', 
                        'crawl_office_v2',
                        'mturk700_none_removed'])
        elif data == 'Tax700V2_1':
            build_taxonomy_impl(taxonomy_folder,
                    data=data,
                    datas=['coco2017', 
                        'voc0712', 
                        'imagenet3k_448', 
                        'imagenet22k_448', 
                        'crawl_office_v1', 
                        'crawl_office_v2',
                        'mturk700_url_as_key'])
        else:
            assert False
    else:
        if 'taxonomy_folder' in param:
            kwargs['taxonomy_folder'] = param['taxonomy_folder']
        #add_reorg = False
    #num_extra_convs = [2]
    num_extra_convs = param['num_extra_convs']
    if len(num_extra_convs) == 1:
        if 'extra_conv_kernel' in param:
            extra_conv_kernel = param['extra_conv_kernel']
        else:
            extra_conv_kernel = [3] * num_extra_convs[0]
    extra_conv_groups = param.get('extra_conv_groups')
    net = param.get('net', 'darknet19_448')
    if data == 'Tax700V1' :
        if not add_reorg and \
                len(num_extra_convs) == 1 and \
                num_extra_convs[0] == 3:
            init_from = {'data': 'office_v2.12', 
                    'net': net, 
                    'expid': 'A_noreorg_burnIn5e.1_tree_initFrom.imagenet.A_bb_nobb'}
        else:
            init_from = {'data': 'imagenet', 'net': 'darknet19_448', 'expid': 'A'}
    else:
        init_from = param['init_from']
    # e.g.
    #for net in ['darknet19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
    #for yolo_coord_scale in [1]:
    yolo_coord_scale = 1
    #for weight_decay in [0.001, 0.005, 0.01, 0.05]:
    weight_decay = 0.0005
    #for net in ['resnet34', 'darknet19_448']:
    #for net in ['resnet101']:
    #for net in ['darknet19_448', 'resnet34', 'resnet101']:
    if yolo_tree:
        kwargs['output_tree_path'] = True

    if net.startswith('resnet'):
        kwargs['cls_add_global_pooling'] = True

    if len(burn_in) > 0 and data != 'voc20':
        kwargs['stageiter'] = ['60e', '90e', '900e'] 
        kwargs['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                batch_size_factor).tolist()
    elif len(burn_in) > 0:
        kwargs['stageiter'] = [5000, 9000, 100000] 
        kwargs['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                batch_size_factor).tolist()
    expid = expid_prefix
    if detmodel == 'yolo':
        expid = expid + ('_noreorg' if not add_reorg else '')
        if len(num_extra_convs) == 0 and num_extra_convs == 0:
            expid = expid + '_noextraconv'
            kwargs['num_extra_convs'] = num_extra_convs
        elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3 \
                or len(num_extra_convs) > 1:
            expid = expid + '_extraconv{}'.format(
                    '_'.join(map(str, num_extra_convs)))
            kwargs['num_extra_convs'] = num_extra_convs
    if multibin_wh:
        expid = expid + '_multibin_wh_{}_{}_{}'.format(multibin_wh_low,
                multibin_wh_high, multibin_wh_count)
        kwargs['multibin_wh'] = multibin_wh
        kwargs['multibin_wh_low'] = multibin_wh_low
        kwargs['multibin_wh_high'] = multibin_wh_high
        kwargs['multibin_wh_count'] = multibin_wh_count
    if not add_reorg and detmodel == 'yolo':
        kwargs['add_reorg'] = add_reorg
    if num_anchor != 5:
        expid = '{}_numAnchor{}'.format(expid, num_anchor)
        assert len(anchor_bias) == 2 * num_anchor
        kwargs['anchor_bias'] = anchor_bias
    if not yolo_rescore:
        expid = '{}_{}'.format(expid, 'norescore')
        kwargs['yolo_rescore'] = yolo_rescore
    if yolo_obj_ignore_center_around:
        expid = '{}_{}'.format(expid, 'ignore')
        kwargs['yolo_obj_ignore_center_around'] = yolo_obj_ignore_center_around
    if yolo_obj_kl_distance:
        expid = '{}_{}'.format(expid, 'objkl')
        kwargs['yolo_obj_kl_distance'] = yolo_obj_kl_distance
    if yolo_xy_kl_distance:
        expid = '{}_xykl'.format(expid)
        kwargs['yolo_xy_kl_distance'] = yolo_xy_kl_distance
    if yolo_obj_only:
        expid = '{}_objonly'.format(expid)
        kwargs['yolo_obj_only'] = yolo_obj_only
    if yolo_exp_linear_wh:
        expid = '{}_explinearwh'.format(expid)
        kwargs['yolo_exp_linear_wh'] = yolo_exp_linear_wh
    if yolo_obj_nonobj_align_to_iou:
        expid = '{}_nonobjtoiou'.format(expid)
        kwargs['yolo_obj_nonobj_align_to_iou'] = yolo_obj_nonobj_align_to_iou
    if yolo_obj_set1_center_around:
        expid = '{}_around1'.format(expid)
        kwargs['yolo_obj_set1_center_around'] = yolo_obj_set1_center_around
    if yolo_obj_nonobj_nopenaltyifsmallthaniou:
        expid = '{}_nolosssmallthaniou'.format(expid)
        kwargs['yolo_obj_nonobj_nopenaltyifsmallthaniou'] = yolo_obj_nonobj_nopenaltyifsmallthaniou
    if yolo_obj_cap_center_around:
        expid = '{}_objcaparound'.format(expid)
        kwargs['yolo_obj_cap_center_around'] = yolo_obj_cap_center_around
    if weight_decay != 0.0005:
        expid = '{}_decay{}'.format(expid, weight_decay)
        kwargs['weight_decay'] = weight_decay
    if yolo_fixed_target:
        expid = '{}_fixedtarget'.format(expid)
        kwargs['yolo_fixed_target'] = yolo_fixed_target
    if yolo_deconv_to_increase_dim:
        expid = '{}_deconvincreasedim'.format(expid)
        kwargs['yolo_deconv_to_increase_dim'] = True
    if yolo_coord_scale != 1:
        expid = '{}_coordscale{}'.format(expid,
                yolo_coord_scale)
        kwargs['yolo_coord_scale'] = yolo_coord_scale
    if not yolo_deconv_to_increase_dim_adapt_bias:
        expid = '{}_nobiaseadapt'.format(expid)
        kwargs['yolo_deconv_to_increase_dim_adapt_bias'] = \
            False
    if yolo_anchor_aligned_images != 12800:
        expid = '{}_align{}'.format(expid,
                yolo_anchor_aligned_images)
        kwargs['yolo_anchor_aligned_images'] = yolo_anchor_aligned_images
    if yolo_nonobj_extra_power != 0:
        expid = '{}_nonobjpower{}'.format(expid,
                yolo_nonobj_extra_power)
        kwargs['yolo_nonobj_extra_power'] = yolo_nonobj_extra_power
    if yolo_obj_extra_power != 0:
        expid = '{}_objpower{}'.format(expid,
                yolo_obj_extra_power)
        kwargs['yolo_obj_extra_power'] = yolo_obj_extra_power
    if yolo_multibin_xy:
        kwargs['yolo_multibin_xy'] = yolo_multibin_xy
        kwargs['yolo_multibin_xy_low'] = yolo_multibin_xy_low
        kwargs['yolo_multibin_xy_high'] = yolo_multibin_xy_high
        kwargs['yolo_multibin_xy_count'] = yolo_multibin_xy_count
        expid = '{}_multibinXY{}'.format(expid,
                yolo_multibin_xy_count)
    if len(dataset_ops) == 1 and \
            dataset_ops[0]['op'] == 'select_top':
        expid = '{}_selectTop{}'.format(expid,
                dataset_ops[0]['num_top'])
    if len(dataset_ops) == 1 and \
            dataset_ops[0]['op'] == 'low_shot':
        low_shot_labels = dataset_ops[0]['labels']
        low_shot_num_train = dataset_ops[0]['num_train']
        expid = '{}_lowShot.{}.{}'.format(expid, 
                low_shot_labels,
                low_shot_num_train)
    if len(dataset_ops) > 0:
        kwargs['dataset_ops'] = dataset_ops
        expid = '{}_dataop{}'.format(expid,
                hash(json.dumps(dataset_ops))%10000)
    if yolo_disable_data_augmentation:
        expid = '{}_noAugmentation'.format(expid)
    if yolo_disable_data_augmentation_except_shift:
        expid = '{}_noAugExpShift'.format(expid)
    if bn_no_train:
        expid = '{}_bnNoTrain'.format(expid)
        kwargs['bn_no_train'] = bn_no_train
    if yolo_object_scale != 5:
        expid = '{}_objscale{}'.format(expid, yolo_object_scale)
        kwargs['yolo_object_scale'] = yolo_object_scale
    if yolo_noobject_scale != 1:
        expid = '{}_noobjScale{}'.format(expid,
                yolo_noobject_scale)
        kwargs['yolo_noobject_scale'] = yolo_noobject_scale
    if yolo_class_scale != 1:
        expid = '{}_clsScale{}'.format(expid,
                yolo_class_scale)
        kwargs['yolo_class_scale'] = yolo_class_scale
    if yolo_avg_replace_max:
        expid = '{}_avgReplaceMax'.format(expid)
        kwargs['yolo_avg_replace_max'] = yolo_avg_replace_max
    if not yolo_sigmoid_xy:
        expid = '{}_nosigmoidXY'.format(expid)
        kwargs['yolo_sigmoid_xy'] = yolo_sigmoid_xy
    if yolo_delta_region3:
        expid = '{}_deltaRegion3'.format(expid)
        kwargs['yolo_delta_region3'] = yolo_delta_region3
    if yolo_background_class:
        expid = '{}_bkgCls{}'.format(expid,
                yolo_use_background_class_to_reduce_obj)
        kwargs['yolo_background_class'] = True
        kwargs['yolo_use_background_class_to_reduce_obj'] = yolo_use_background_class_to_reduce_obj
        if yolo_iou_th_to_use_bkg_cls != 1:
            expid = '{}_iouTh{}'.format(expid,
                    yolo_iou_th_to_use_bkg_cls)
            kwargs['yolo_iou_th_to_use_bkg_cls'] = yolo_iou_th_to_use_bkg_cls
    if res_loss:
        expid = '{}_resLoss'.format(expid)
        kwargs['res_loss'] = res_loss
        kwargs['skip_genprototxt'] = True
    if yolo_per_class_obj:
        expid = '{}_perClassObj'.format(expid)
        kwargs['yolo_per_class_obj'] = yolo_per_class_obj
    if not last_conv_bias:
        expid = '{}_noBiasLastConv'.format(expid)
        kwargs['yolo_last_conv_bias'] = last_conv_bias
    if yolo_low_shot_regularizer:
        expid = '{}_lowShotEqualNorm'.format(expid)
        kwargs['yolo_low_shot_regularizer'] = True
    if detmodel == 'yolo' and yolo_full_gpu:
        kwargs['yolo_full_gpu'] = yolo_full_gpu
        if expid_prefix == 'A':
            expid = '{}_fullGpu'.format(expid)
    if burn_in != '':
        expid = '{}_burnIn{}.{}'.format(expid, burn_in,
                burn_in_power)
        kwargs['burn_in'] = burn_in
        kwargs['burn_in_power'] = burn_in_power
    if len(rotate_max) != 1 or len(rotate_max[0]) != 1 or \
            rotate_max[0][0] != 0:
        kwargs['rotate_max'] = rotate_max
        expid = '{}_rotate{}'.format(expid, '.'.join([str(y) 
            for x in rotate_max for y in x]))
    if len(rotate_with_90) != 1 or \
            len(rotate_with_90[0]) != 1 or \
            rotate_with_90[0][0]:
        expid = '{}_r90.{}'.format(expid, '.'.join(['1' if x else '0' for y in
                rotate_with_90 for x in y]))
        kwargs['rotate_with_90'] = rotate_with_90
    if len(box_data_param_weightss) != 1 or \
            len(box_data_param_weightss[0]) != 1 or \
            box_data_param_weightss[0][0] != 1:
        kwargs['box_data_param_weightss'] = box_data_param_weightss
        expid = '{}_boxWeight.{}'.format(expid, '.'.join([str(y) for x in
                box_data_param_weightss for y in x]))
    #if rotate_max != 0:
        #kwargs['rotate_max'] = rotate_max
        #expid = '{}_rotate{}'.format(expid, rotate_max)
    #if rotate_with_90:
        #expid = '{}_with90'.format(expid)
        #kwargs['rotate_with_90'] = True
    if yolo_random_scale_min != 0.25:
        expid = '{}_randomScaleMin{}'.format(expid, 
                yolo_random_scale_min)
        kwargs['yolo_random_scale_min'] = yolo_random_scale_min
    if yolo_random_scale_max != 2:
        expid = '{}_randomScaleMax{}'.format(expid,
                yolo_random_scale_max)
        kwargs['yolo_random_scale_max'] = yolo_random_scale_max
    if scale_relative_input:
        expid = '{}_RelativeScale2'.format(expid)
        kwargs['scale_relative_input'] = scale_relative_input
    if nms_type != 'Standard' and nms_type != '':
        if nms_type == 'LinearSoft':
            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.LinearSoft 
        if nms_type == 'GaussianSoft':
            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.GaussianSoft
        if gaussian_nms_sigma != 0.5:
            kwargs['gaussian_nms_sigma'] = 0.5
    if incorporate_at_least_one_box:
        expid = '{}_atLeastOneBB'.format(expid)
        kwargs['incorporate_at_least_one_box'] = incorporate_at_least_one_box
    if scale_constrained_by_one_box_area:
        expid = '{}_scaleConstrainedByOne'.format(expid)
        kwargs['scale_constrained_by_one_box_area'] = scale_constrained_by_one_box_area
        if scale_constrained_by_one_box_area_min != 0.001:
            expid = '{}_Min{}'.format(expid, scale_constrained_by_one_box_area_min)
            kwargs['scale_constrained_by_one_box_area_min'] = scale_constrained_by_one_box_area_min
    if yolo_tree and detmodel == 'yolo':
        if 'TaxVocPerson' not in data:
            expid = '{}_tree'.format(expid)
        kwargs['yolo_tree'] = yolo_tree
        if test_tree_cls_specific_th_by_average is not None:
            kwargs['test_tree_cls_specific_th_by_average'] = test_tree_cls_specific_th_by_average
        kwargs['yolo_tree_eval_label_lift'] = False
    if len(init_from) > 0 or 'basemodel' in param:
        if len(init_from) > 0:
            assert net == init_from['net']
            c = CaffeWrapper(data=init_from['data'], 
                    net=init_from['net'],
                    expid=init_from['expid'])
            kwargs['basemodel'] = c.best_model().model_param
        else:
            kwargs['basemodel'] = param['basemodel']
        expid = '{}_init{}'.format(expid,
                hash(kwargs['basemodel']) % 10000)
    if yolo_angular_loss:
        expid = '{}_AngularRegulizer'.format(expid)
        kwargs['yolo_angular_loss'] = True
        if yolo_angular_loss_weight != 1:
            expid = '{}Weight{}'.format(expid,
                    yolo_angular_loss_weight)
            kwargs['yolo_angular_loss_weight'] = yolo_angular_loss_weight
    if len(no_bias) > 0:
        expid = '{}_noBias{}'.format(expid, no_bias)
        kwargs['no_bias'] = no_bias
    if net_input_size_min != 416:
        expid = '{}_InMin{}'.format(expid, net_input_size_min)
        kwargs['net_input_size_min'] = net_input_size_min
    if net_input_size_max != 416:
        expid = '{}_InMax{}'.format(expid, net_input_size_max)
        kwargs['net_input_size_max'] = net_input_size_max
    if any(k != 3 for k in extra_conv_kernel):
        expid = '{}_extraConvKernel.{}'.format(expid, '.'.join(
            map(str, extra_conv_kernel)))
        kwargs['extra_conv_kernel'] = extra_conv_kernel
    if any(c != 1024 for c in extra_conv_channels):
        expid = '{}_extraChannels.{}'.format(expid,
                '.'.join(map(str, extra_conv_channels)))
        kwargs['extra_conv_channels'] = extra_conv_channels
    if len(last_fixed_param) > 0:
        expid = '{}_FixParam.{}'.format(expid,
                last_fixed_param.replace('/', '.'))
        kwargs['last_fixed_param'] = last_fixed_param
    if residual_loss:
        expid = '{}_resLoss{}'.format(expid, 
                '.'.join(map(lambda x: x.replace('/', '_'),
                    residual_loss_froms)))
        kwargs['residual_loss'] = residual_loss
        kwargs['residual_loss_froms'] = residual_loss_froms
    if yolo_disable_no_penalize_if_iou_large:
        kwargs['yolo_disable_no_penalize_if_iou_large'] = True
        expid = '{}_disableNoPenIfIouLarge'.format(expid)
    if cutout_prob > 0:
        kwargs['cutout_prob'] = cutout_prob
        expid = '{}_cutoutProb{}'.format(expid, cutout_prob)
    if len(multi_feat_anchor) > 0:
        kwargs['multi_feat_anchor'] = multi_feat_anchor
        expid = '{}_multiFeatAnchor{}.{}.{}'.format(expid,
                len(multi_feat_anchor),
                '.'.join(map(str, [m['loss_weight_multiplier'] for m in
                    multi_feat_anchor])),
                hash(json.dumps(multi_feat_anchor))%10000)
    if detmodel == 'yolo':
        if max_iters != '128e' and max_iters != 10000:
            expid = '{}_maxIter.{}'.format(expid, max_iters)
    if detmodel == 'yolo' and yolo_softmax_norm_by_valid:
        kwargs['yolo_softmax_norm_by_valid'] = yolo_softmax_norm_by_valid
        if expid_prefix == 'A':
            expid = '{}_softmaxByValid'.format(expid)
    if yolo_softmax_extra_weight != 1 and expid_prefix == 'A':
        expid = '{}_softmaxWeight{}'.format(expid,
                yolo_softmax_extra_weight)
        #kwargs['yolo_softmax_extra_weight'] = yolo_softmax_extra_weight
    if num_bn_fix > 0:
        expid = '{}_BNFix{}'.format(expid, num_bn_fix)
        kwargs['num_bn_fix'] = num_bn_fix
    if ignore_negative_first_batch:
        expid = '{}_ignoreNegativeFirst'.format(expid)
        kwargs['ignore_negative_first_batch'] = ignore_negative_first_batch
        if yolo_not_ignore_negative_seen_images > 0:
            expid = '{}_notIgnore{}'.format(expid,
                    yolo_not_ignore_negative_seen_images)
            kwargs['yolo_not_ignore_negative_seen_images'] = yolo_not_ignore_negative_seen_images
        if yolo_force_negative_with_partial_overlap:
            expid = '{}_ForcePartial'.format(expid)
            kwargs['yolo_force_negative_with_partial_overlap'] = True
    if yolo_xywh_norm_by_weight_sum:
        expid = '{}_xywhNormWeight'.format(expid)
        kwargs['yolo_xywh_norm_by_weight_sum'] = True
    if detmodel == 'yolo':
        if len(test_input_sizes) != 1 or test_input_sizes[0] != 416:
            kwargs['test_input_sizes'] = test_input_sizes
    if test_data != data:
        kwargs['test_data'] = test_data
    if yolo_index_threshold_loss_extra_weight != 1:
        kwargs['yolo_index_threshold_loss_extra_weight'] = yolo_index_threshold_loss_extra_weight
        expid = '{}_IndexLossWeight{}'.format(expid, yolo_index_threshold_loss_extra_weight)
    if not all(t == 1 for s in tsv_box_max_samples for t in s):
        kwargs['tsv_box_max_samples'] = tsv_box_max_samples
        expid = '{}_TsvBoxSamples{}'.format(expid, '.'.join([str(t) for s in tsv_box_max_samples
                for t in s]))
    if first_batch_objectiveness_enhancement:
        expid = '{}_FirstObjEnhance{}'.format(expid,
                first_batch_objectiveness_enhancement_weight)
        kwargs['first_batch_objectiveness_enhancement'] = True
        kwargs['first_batch_objectiveness_enhancement_weight'] = first_batch_objectiveness_enhancement_weight
    if extra_conv_groups is not None:
        expid = '{}ExtraConvGroups{}'.format(expid, '_'.join(map(str,
            extra_conv_groups)))
        kwargs['extra_conv_groups'] = extra_conv_groups
    if param.get('softmax_tree_prediction_threshold', 0.5) != 0.5:
        kwargs['softmax_tree_prediction_threshold'] = param['softmax_tree_prediction_threshold']
    if param.get('skip_genprototxt', False):
        kwargs['skip_genprototxt'] = True
    if param.get('load_parameter', False):
        kwargs['load_parameter'] = True
    if param.get('test_split', '') != '':
        kwargs['test_split'] = param['test_split']
    if data == 'imagenet2012':
        kwargs['predict_style'] = 'tsvdatalayer'
    if detmodel == 'classification':
        if param['use_tsvbox_for_cls']:
            kwargs['use_tsvbox_for_cls'] = True
            expid = '{}TsvBox'.format(expid)
    expid = expid + param['suffix']
    kwargs['monitor_train_only'] = monitor_train_only
    kwargs['expid'] = expid
    kwargs['net'] = net
    kwargs['data'] = data
    all_task.append(kwargs)
    
    if len(all_task) == 1 and \
            (all_task[0]['data'].startswith('Tax') or all_task[0]['data'] ==
                    'imagenet2012'):
        if not monitor_train_only:
            all_resource = get_all_resources(num_gpu=8)
    else:
        all_resource = get_all_resources()
    if monitor_train_only:
        all_resource = get_all_resources(exclude=['djx'], num_gpu=2)
    all_resource = get_all_resources(num_gpu=8)
    logging.info(pformat(all_resource))
    logging.info('#resource: {}'.format(len(all_resource)))
    logging.info('#task: {}'.format(len(all_task)))

    task_type = param['task_type']
    if task_type == 'speed':
        task = all_task[0]
        c = CaffeWrapper(**task)
        c._ensure_macc_calculated()
        c.cpu_test_time()
        c.gpu_test_time()
    elif task_type == 'batch_run':
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
    elif task_type.startswith('debug'):
        idx = -1
        task = all_task[idx]
        if task_type == 'debug_train':
            task['effective_batch_size'] = 8
            task['force_train'] = True
            task['debug_train'] = True
        elif task_type == 'debug_predict':
            all_task[idx]['debug_detect'] = True
            all_task[idx]['force_predict'] = True
        elif task_type == 'debug_evaluate':
            task['force_evaluate'] = True
        task_processor(({}, range(1)), task)
    elif task_type == 'demo':
        pass
        #c.demo(None)
        #rows = tsv_reader(tsv_file)
        #for row in rows:
            #continue
        #import ipdb;ipdb.set_trace()
        #c.demo('./data/office100_v1_with_bb/train.tsv')
        #c.demo(tsv_file)
        #c.demo('/raid/jianfw/data/office100_crawl/TermList.instagram.pinterest.scrapping.image.tsv')
        #c.demo('/raid/jianfw/work/yuxiao_crop/ring/')
        #c.demo('tmp.png')
        #all_task[0]['force_predict'] = True
        #return
    return all_task

def officev2_11():
    all_task = []
    max_num = 500
    all_task = []
    machine_ips = []
    #for monitor_train_only in [True]:
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    num_extra_convs = 3
    #suffix = '_1'
    suffix = '_2gpu'
    #suffix = '_xx'
    #dataset_ops_template = []
    #dataset_ops = []
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    batch_size_factor = 2
    suffix = '_batchSizeFactor{}'.format(batch_size_factor) \
            if batch_size_factor != 1 else ''
    #suffix = '{}{}'.format(suffix, '_256e')
    suffix = '_withNoBB' 
    suffix = '' 
    effective_batch_size = 64 * batch_size_factor
    #max_iters=1
    def gen_anchor_bias(num_anchor):
        if num_anchor == 0:
            return None
        if num_anchor == 2:
            return [4, 8, 8, 4]
        result = []
        n = int(np.sqrt(num_anchor))
        assert n * n == num_anchor
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 9
    anchor_bias = gen_anchor_bias(num_anchor)
    anchor_bias = None
    num_anchor = 5
    multibin_wh = False
    multibin_wh_low = 0;
    multibin_wh_high = 13;
    all_multibin_wh_count = [16]
    #all_multibin_wh_count = [16, 32, 48, 64];
    yolo_rescore = True
    yolo_xy_kl_distance = False
    yolo_obj_only = False
    yolo_exp_linear_wh = False
    yolo_obj_nonobj_align_to_iou = False
    yolo_obj_ignore_center_around = False
    yolo_obj_kl_distance = False
    yolo_obj_set1_center_around = False
    yolo_blame = 'xy.wh.obj.cls.nonobj'
    yolo_blame = 'xy.wh.obj.nonobj.cls'
    yolo_blame = ''
    yolo_deconv_to_increase_dim = False
    yolo_deconv_to_increase_dim_adapt_bias = True
    yolo_anchor_aligned_images = 1280000000
    yolo_anchor_aligned_images = 12800
    #yolo_anchor_aligned_images = 0
    yolo_nonobj_extra_power = 0
    yolo_obj_extra_power = 0
    yolo_disable_data_augmentation = False
    yolo_disable_data_augmentation_except_shift = False
    #yolo_anchor_aligned_images = 0
    bn_no_train = False
    yolo_coords_only = False
    if yolo_coords_only:
        yolo_object_scale = 0
        yolo_noobject_scale = 0
        yolo_class_scale = 0
    else:
        yolo_object_scale = 5
        yolo_noobject_scale = 1
        yolo_class_scale = 1
    yolo_avg_replace_max = False
    yolo_per_class_obj = False
    max_iters = '128e'
    max_iters = 5
    #data = 'voc20'
    #data = 'brand1048'
    #data = 'office100_v1'
    #max_iters=10000 / batch_size_factor
    burn_in = '5e'
    burn_in = ''
    burn_in_power = 1

    yolo_full_gpu = True
    #yolo_full_gpu = False
    test_tree_cls_specific_th_by_average = 1.2
    test_tree_cls_specific_th_by_average = None
    yolo_angular_loss = True
    yolo_angular_loss = False
    net_input_size_min = 416
    net_input_size_max = 416
    no_bias = 'conf'
    no_bias = ''
    init_from = {'data': 'imagenet', 'net': 'darknet19_448', 'expid': 'A'}
    #init_from = {'data': 'office_v2.1', 'net': 'darknet19_448', 
        #'expid': 'A_burnIn5e.1_tree_initFrom.imagenet.A'}
    #init_from = {}
    kwargs_template = dict(
            detmodel='yolo',
            max_iters=max_iters,
            #max_iters='80e',
            #max_iters='256e',
            #max_iters=max_iters,
            #max_iters=11000,
            #yolo_blame=yolo_blame,
            #expid=expid,
            #yolo_jitter=0,
            #yolo_hue=0,
            #yolo_test_fix_xy = True,
            #yolo_test_fix_wh = True,
            #yolo_extract_target_prediction = True,
            #yolo_max_truth=300,
            #yolo_exposure=1,
            #test_on_train = True,
            #yolo_saturation=1,
            #yolo_random_scale_min=1,
            #yolo_random_scale_max=1,
            #expid='A_multibin_wh_0_13_16_no2-wh',
            #expid='baseline_2',
            #snapshot=1000,
            #snapshot=0,
            #target_synset_tree='./aux_data/yolo/9k.tree',
            #target_synset_tree='./data/{}/tree.txt'.format(data),
            #dataset_ops=dataset_ops,
            #effective_batch_size=1,
            #num_anchor=3,
            #num_anchor=num_anchor,
            #force_train=True,
            #force_evaluate=True,
            #debug_detect=True,
            #force_predict=True,
            #extract_features='angular_loss.softmax_loss.o_obj_loss.xy_loss.wh_loss.o_noobj_loss',
            #data_dependent_init=True,
            #restore_snapshot_iter=-1,
            #display=0,
            #region_debug_info=10,
            #display=100,
            #stagelr=stagelr,
            #anchor_bias=anchor_bias,
            #test_input_sizes=[288, 416, 480, 608],
            #test_input_sizes=[416, 608],
            #test_input_sizes=[608, 416],
            #stageiter=[1000, 1000000],
            #stageiter=[100,5000,9000,10000000],
            #stageiter = (np.asarray([5000, 9000, 1000000]) / batch_size_factor).tolist(),
            #stagelr = (np.asarray([0.001,0.0001,0.00001]) * batch_size_factor).tolist(),
            #stagelr=[0.0001,0.001,0.0001,0.0001],
            #burn_in=100,
            #class_specific_nms=False,
            #basemodel='./output/imagenet_darknet19_448_A/snapshot/model_iter_570640.caffemodel',
            #effective_batch_size=effective_batch_size,
            #solver_debug_info=True,
            yolo_test_maintain_ratio = True,
            ovthresh = [0,0.1,0.2,0.3,0.4,0.5])

    if effective_batch_size != 64:
        kwargs_template['effective_batch_size'] = effective_batch_size

    if yolo_disable_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
        kwargs_template['yolo_fix_offset'] = True
        kwargs_template['yolo_mirror'] = False
    elif yolo_disable_data_augmentation_except_shift:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1

    continue_less_data_augmentation = False
    if continue_less_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
    if batch_size_factor == 2:
        burn_in = '5e'
        burn_in_power = 1

    #adv = [(False, True), (True, False), (True, True)]
    #max_iters = 4
    #for multibin_wh_count in all_multibin_wh_count:
    multibin_wh_count = all_multibin_wh_count[0]
    #for yolo_obj_ignore_center_around, yolo_obj_kl_distance in adv:
    yolo_fixed_target = False
    yolo_obj_nonobj_nopenaltyifsmallthaniou = False
    yolo_obj_cap_center_around = False
    yolo_multibin_xy = False
    yolo_sigmoid_xy = True
    yolo_delta_region3 = False
    yolo_background_class = False
    #yolo_background_class = False
    yolo_use_background_class_to_reduce_obj = 0.4
    #for multibin_xy_count in [32, 16, 8]:
    multibin_xy_count = 4
    #res_loss = True
    res_loss = False
    yolo_use_background_class_to_reduce_obj = 1
    monitor_train_only = False
    #monitor_train_only = True
    #for yolo_use_background_class_to_reduce_obj in [1, 0.8, 0.6, 0.4, 0.2]:
    yolo_low_shot_regularizer = False
    #full_labels = dataset.load_labelmap()
    full_labels = []
    #for low_shot_label in full_labels:
    yolo_random_scale_max = 2
    scale_relative_input = True
    scale_relative_input = False
    nms_type = 'LinearSoft'
    nms_type = 'GaussianSoft'
    nms_type = ''
    gaussian_nms_sigma = 0.5
    scale_constrained_by_one_box_area_min = 0.001
    last_fixed_param = 'dark5e/leaky'
    last_fixed_param = ''
    #for low_shot_label in ['']:
    #for low_shot_label in [full_labels[0]]:
    full_labels.insert(0, '')
    #for low_shot_label in full_labels[:5]:
    low_shot_label = ''
    #for low_shot_label in ['']:
    #for extra_conv_kernel in [[1, 1], [1, 3], [3, 1]]:
    for extra_conv_kernel in [[3, 3, 3]]:
    #for extra_conv_channels in [[1024, 512, 512], [512, 512, 1024], [512, 1024,
        #512]]:
        extra_conv_channels = [1024, 1024, 1024]
    #for extra_conv_channels in [[1024, 1024]]:
    #for extra_conv_channels in [[1024, 1024, 1024]]:
    #for extra_conv_kernel in [[3,3,3], [1, 1, 1], [1, 3, 1], [3, 1, 1], [1, 1, 3], [3,
        #3, 1], [3, 1, 3], [1, 3, 3]]:
    #for extra_conv_kernel in [[1, 3, 1]]:
        if len(low_shot_label) > 0:
            dataset_ops = [{'op':'low_shot', 'labels': 'dog', 'num_train': 1}]
            dataset_ops[0]['labels'] = low_shot_label
            dataset_ops[0]['labels_idx'] = full_labels.index(low_shot_label)
        else:
            dataset_ops = [{'op':'remove'},
                    {'op':'add',
                     'name':'office_v2.11_with_bb',
                     'source':'train',
                     'weight': 3},
                    #{'op': 'add',
                     #'name': 'office_v2.1_no_bb',
                     #'source': 'train',
                     #'weight': 1},
                    ]
            dataset_ops = []
            #for multibin_xy_count in [4]:
        yolo_multibin_xy_low = 0.5 / multibin_xy_count
        #multibin_xy_count = 16
        yolo_multibin_xy_high = 1 - 0.5 / multibin_xy_count
        yolo_multibin_xy_count = multibin_xy_count
        #for yolo_obj_kl_distance in [False]:
        yolo_obj_kl_distance = False
    #for yolo_object_scale in [40, 60]:
        #for monitor_train_only in [False, True]:
        #for monitor_train_only in [True]:
        #for yolo_iou_th_to_use_bkg_cls in [0.1]:
        yolo_iou_th_to_use_bkg_cls = 0.1
        #for last_conv_bias in [True]:
        last_conv_bias = True
        #for rotate_max in [5, 10, 15, 20]:
        rotate_max = 10
        rotate_max = 0
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, True), (True, False)]:
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, False)]:
        incorporate_at_least_one_box, scale_constrained_by_one_box_area = False, False
        #for yolo_angular_loss_weight in [0.1, 1, 10]:
        #for yolo_angular_loss_weight in [1]:
        yolo_angular_loss_weight = 1
        #for data in ['fridge_clean', 'voc20']:
        #for data in ['office_v2.12']:
        for data in ['office_debug']:
        #for data in ['voc20']:
            if data.startswith('office'):
                yolo_tree = True
            else:
                yolo_tree = False
            if test_tree_cls_specific_th_by_average:
                assert yolo_tree
        #rotate_max = 10
        #rotate_max = 0
            rotate_with_90 = False
        #for yolo_random_scale_min in [0.5, 0.75]:
            yolo_random_scale_min = 0.25
            yolo_random_scale_max = 2
        #for monitor_train_only in [False]:
        #monitor_train_only = True
        #monitor_train_only = True
    #for monitor_train_only in [True]:
    #for monitor_train_only in [False, True]:
            #for add_reorg, num_extra_convs in [(True, [6]), (True, [7])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(False, [3]), (True, [3])]:
            for add_reorg, num_extra_convs in [(False, [3])]:
        #for add_reorg, num_extra_convs in all_extra_option:
        #for weight_decay in [0.0001, 0.00005]:
        #for weight_decay in [0.0005]:
            #weight_decay = 0.0005
            #weight_decay = 0
            #for net in ['darknet19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
            #for net in ['resnet34']:
            #net = 'darknet19'
            #for net in ['resnet101']:
            #for yolo_coord_scale in [1]:
                yolo_coord_scale = 1
                #for data in ['voc20', 'voc2012', 'coco2017', 'imagenet']:
                #for data in ['coco2017']:
                #for weight_decay in [0.001, 0.005, 0.01, 0.05]:
                weight_decay = 0.0005
                #for net in ['resnet34', 'darknet19_448']:
                for net in ['darknet19_448']:
                #for net in ['resnet34']:
                    if len(burn_in) > 0 and data != 'voc20':
                        kwargs_template['stageiter'] = ['60e', '90e', '900e'] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()
                    elif len(burn_in) > 0:
                        kwargs_template['stageiter'] = [5000, 9000, 100000] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()

                    kwargs = copy.deepcopy(kwargs_template)
                    expid = 'A'
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if len(num_extra_convs) == 0 and num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                        kwargs['num_extra_convs'] = num_extra_convs
                    elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3 \
                            or len(num_extra_convs) > 1:
                        expid = expid + '_extraconv{}'.format(
                                '_'.join(map(str, num_extra_convs)))
                        kwargs['num_extra_convs'] = num_extra_convs
                    if multibin_wh:
                        expid = expid + '_multibin_wh_{}_{}_{}'.format(multibin_wh_low,
                                multibin_wh_high, multibin_wh_count)
                        kwargs['multibin_wh'] = multibin_wh
                        kwargs['multibin_wh_low'] = multibin_wh_low
                        kwargs['multibin_wh_high'] = multibin_wh_high
                        kwargs['multibin_wh_count'] = multibin_wh_count
                    if not add_reorg:
                        kwargs['add_reorg'] = add_reorg
                    if num_anchor != 5:
                        expid = '{}_numAnchor{}'.format(expid, num_anchor)
                        assert len(anchor_bias) == 2 * num_anchor
                        kwargs['anchor_bias'] = anchor_bias
                    if not yolo_rescore:
                        expid = '{}_{}'.format(expid, 'norescore')
                        kwargs['yolo_rescore'] = yolo_rescore
                    if yolo_obj_ignore_center_around:
                        expid = '{}_{}'.format(expid, 'ignore')
                        kwargs['yolo_obj_ignore_center_around'] = yolo_obj_ignore_center_around
                    if yolo_obj_kl_distance:
                        expid = '{}_{}'.format(expid, 'objkl')
                        kwargs['yolo_obj_kl_distance'] = yolo_obj_kl_distance
                    if yolo_xy_kl_distance:
                        expid = '{}_xykl'.format(expid)
                        kwargs['yolo_xy_kl_distance'] = yolo_xy_kl_distance
                    if yolo_obj_only:
                        expid = '{}_objonly'.format(expid)
                        kwargs['yolo_obj_only'] = yolo_obj_only
                    if yolo_exp_linear_wh:
                        expid = '{}_explinearwh'.format(expid)
                        kwargs['yolo_exp_linear_wh'] = yolo_exp_linear_wh
                    if yolo_obj_nonobj_align_to_iou:
                        expid = '{}_nonobjtoiou'.format(expid)
                        kwargs['yolo_obj_nonobj_align_to_iou'] = yolo_obj_nonobj_align_to_iou
                    if yolo_obj_set1_center_around:
                        expid = '{}_around1'.format(expid)
                        kwargs['yolo_obj_set1_center_around'] = yolo_obj_set1_center_around
                    if yolo_obj_nonobj_nopenaltyifsmallthaniou:
                        expid = '{}_nolosssmallthaniou'.format(expid)
                        kwargs['yolo_obj_nonobj_nopenaltyifsmallthaniou'] = yolo_obj_nonobj_nopenaltyifsmallthaniou
                    if yolo_obj_cap_center_around:
                        expid = '{}_objcaparound'.format(expid)
                        kwargs['yolo_obj_cap_center_around'] = yolo_obj_cap_center_around
                    if weight_decay != 0.0005:
                        expid = '{}_decay{}'.format(expid, weight_decay)
                        kwargs['weight_decay'] = weight_decay
                    if yolo_fixed_target:
                        expid = '{}_fixedtarget'.format(expid)
                        kwargs['yolo_fixed_target'] = yolo_fixed_target
                    if yolo_deconv_to_increase_dim:
                        expid = '{}_deconvincreasedim'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim'] = True
                    if yolo_coord_scale != 1:
                        expid = '{}_coordscale{}'.format(expid,
                                yolo_coord_scale)
                        kwargs['yolo_coord_scale'] = yolo_coord_scale
                    if not yolo_deconv_to_increase_dim_adapt_bias:
                        expid = '{}_nobiaseadapt'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim_adapt_bias'] = \
                            False
                    if yolo_anchor_aligned_images != 12800:
                        expid = '{}_align{}'.format(expid,
                                yolo_anchor_aligned_images)
                        kwargs['yolo_anchor_aligned_images'] = yolo_anchor_aligned_images
                    if yolo_nonobj_extra_power != 0:
                        expid = '{}_nonobjpower{}'.format(expid,
                                yolo_nonobj_extra_power)
                        kwargs['yolo_nonobj_extra_power'] = yolo_nonobj_extra_power
                    if yolo_obj_extra_power != 0:
                        expid = '{}_objpower{}'.format(expid,
                                yolo_obj_extra_power)
                        kwargs['yolo_obj_extra_power'] = yolo_obj_extra_power
                    if yolo_multibin_xy:
                        kwargs['yolo_multibin_xy'] = yolo_multibin_xy
                        kwargs['yolo_multibin_xy_low'] = yolo_multibin_xy_low
                        kwargs['yolo_multibin_xy_high'] = yolo_multibin_xy_high
                        kwargs['yolo_multibin_xy_count'] = yolo_multibin_xy_count
                        expid = '{}_multibinXY{}'.format(expid,
                                yolo_multibin_xy_count)
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'select_top':
                        expid = '{}_selectTop{}'.format(expid,
                                dataset_ops[0]['num_top'])
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'low_shot':
                        low_shot_labels = dataset_ops[0]['labels']
                        low_shot_num_train = dataset_ops[0]['num_train']
                        expid = '{}_lowShot.{}.{}'.format(expid, 
                                low_shot_labels,
                                low_shot_num_train)
                    if len(dataset_ops) > 0:
                        kwargs['dataset_ops'] = dataset_ops
                    if yolo_disable_data_augmentation:
                        expid = '{}_noAugmentation'.format(expid)
                    if yolo_disable_data_augmentation_except_shift:
                        expid = '{}_noAugExpShift'.format(expid)
                    if bn_no_train:
                        expid = '{}_bnNoTrain'.format(expid)
                        kwargs['bn_no_train'] = bn_no_train
                    if yolo_object_scale != 5:
                        expid = '{}_objscale{}'.format(expid, yolo_object_scale)
                        kwargs['yolo_object_scale'] = yolo_object_scale
                    if yolo_noobject_scale != 1:
                        expid = '{}_noobjScale{}'.format(expid,
                                yolo_noobject_scale)
                        kwargs['yolo_noobject_scale'] = yolo_noobject_scale
                    if yolo_class_scale != 1:
                        expid = '{}_clsScale{}'.format(expid,
                                yolo_class_scale)
                        kwargs['yolo_class_scale'] = yolo_class_scale
                    if yolo_avg_replace_max:
                        expid = '{}_avgReplaceMax'.format(expid)
                        kwargs['yolo_avg_replace_max'] = yolo_avg_replace_max
                    if not yolo_sigmoid_xy:
                        expid = '{}_nosigmoidXY'.format(expid)
                        kwargs['yolo_sigmoid_xy'] = yolo_sigmoid_xy
                    if yolo_delta_region3:
                        expid = '{}_deltaRegion3'.format(expid)
                        kwargs['yolo_delta_region3'] = yolo_delta_region3
                    if yolo_background_class:
                        expid = '{}_bkgCls{}'.format(expid,
                                yolo_use_background_class_to_reduce_obj)
                        kwargs['yolo_background_class'] = True
                        kwargs['yolo_use_background_class_to_reduce_obj'] = yolo_use_background_class_to_reduce_obj
                        if yolo_iou_th_to_use_bkg_cls != 1:
                            expid = '{}_iouTh{}'.format(expid,
                                    yolo_iou_th_to_use_bkg_cls)
                            kwargs['yolo_iou_th_to_use_bkg_cls'] = yolo_iou_th_to_use_bkg_cls
                    if res_loss:
                        expid = '{}_resLoss'.format(expid)
                        kwargs['res_loss'] = res_loss
                        kwargs['skip_genprototxt'] = True
                    if yolo_per_class_obj:
                        expid = '{}_perClassObj'.format(expid)
                        kwargs['yolo_per_class_obj'] = yolo_per_class_obj
                    if not last_conv_bias:
                        expid = '{}_noBiasLastConv'.format(expid)
                        kwargs['yolo_last_conv_bias'] = last_conv_bias
                    if yolo_low_shot_regularizer:
                        expid = '{}_lowShotEqualNorm'.format(expid)
                        kwargs['yolo_low_shot_regularizer'] = True
                    if yolo_full_gpu:
                        expid = '{}_fullGpu'.format(expid)
                        kwargs['yolo_full_gpu'] = yolo_full_gpu
                    if burn_in != '':
                        expid = '{}_burnIn{}.{}'.format(expid, burn_in,
                                burn_in_power)
                        kwargs['burn_in'] = burn_in
                        kwargs['burn_in_power'] = burn_in_power
                    if rotate_max != 0:
                        kwargs['rotate_max'] = rotate_max
                        expid = '{}_rotate{}'.format(expid, rotate_max)
                    if rotate_with_90:
                        expid = '{}_with90'.format(expid)
                        kwargs['rotate_with_90'] = True
                    if yolo_random_scale_min != 0.25:
                        expid = '{}_randomScaleMin{}'.format(expid, 
                                yolo_random_scale_min)
                        kwargs['yolo_random_scale_min'] = yolo_random_scale_min
                    if yolo_random_scale_max != 2:
                        expid = '{}_randomScaleMax{}'.format(expid,
                                yolo_random_scale_max)
                        kwargs['yolo_random_scale_max'] = yolo_random_scale_max
                    if scale_relative_input:
                        expid = '{}_RelativeScale2'.format(expid)
                        kwargs['scale_relative_input'] = scale_relative_input
                    if nms_type != 'Standard' and nms_type != '':
                        if nms_type == 'LinearSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.LinearSoft 
                        if nms_type == 'GaussianSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.GaussianSoft
                        if gaussian_nms_sigma != 0.5:
                            kwargs['gaussian_nms_sigma'] = 0.5
                    if incorporate_at_least_one_box:
                        expid = '{}_atLeastOneBB'.format(expid)
                        kwargs['incorporate_at_least_one_box'] = incorporate_at_least_one_box
                    if scale_constrained_by_one_box_area:
                        expid = '{}_scaleConstrainedByOne'.format(expid)
                        kwargs['scale_constrained_by_one_box_area'] = scale_constrained_by_one_box_area
                        if scale_constrained_by_one_box_area_min != 0.001:
                            expid = '{}_Min{}'.format(expid, scale_constrained_by_one_box_area_min)
                            kwargs['scale_constrained_by_one_box_area_min'] = scale_constrained_by_one_box_area_min
                    if yolo_tree:
                        expid = '{}_tree'.format(expid)
                        kwargs['yolo_tree'] = yolo_tree
                        if test_tree_cls_specific_th_by_average is not None:
                            kwargs['test_tree_cls_specific_th_by_average'] = test_tree_cls_specific_th_by_average
                    if len(init_from) > 0:
                        assert net == init_from['net']
                        if len(init_from['expid']) > 5:
                            expid = '{}_initFrom'.format(expid)
                        else:
                            expid = '{}_initFrom.{}.{}'.format(expid,
                                    init_from['data'], init_from['expid'])
                        c = CaffeWrapper(data=init_from['data'], 
                                net=init_from['net'],
                                expid=init_from['expid'])
                        kwargs['basemodel'] = c.best_model().model_param
                    if yolo_angular_loss:
                        expid = '{}_AngularRegulizer'.format(expid)
                        kwargs['yolo_angular_loss'] = True
                        if yolo_angular_loss_weight != 1:
                            expid = '{}Weight{}'.format(expid,
                                    yolo_angular_loss_weight)
                            kwargs['yolo_angular_loss_weight'] = yolo_angular_loss_weight
                    if len(no_bias) > 0:
                        expid = '{}_noBias{}'.format(expid, no_bias)
                        kwargs['no_bias'] = no_bias
                    if net_input_size_min != 416:
                        expid = '{}_InMin{}'.format(expid, net_input_size_min)
                        kwargs['net_input_size_min'] = net_input_size_min
                    if net_input_size_max != 416:
                        expid = '{}_InMax{}'.format(expid, net_input_size_max)
                        kwargs['net_input_size_max'] = net_input_size_max
                    if any(k != 3 for k in extra_conv_kernel):
                        expid = '{}_extraConvKernel.{}'.format(expid, '.'.join(
                            map(str, extra_conv_kernel)))
                        kwargs['extra_conv_kernel'] = extra_conv_kernel
                    if any(c != 1024 for c in extra_conv_channels):
                        expid = '{}_extraChannels.{}'.format(expid,
                                '.'.join(map(str, extra_conv_channels)))
                        kwargs['extra_conv_channels'] = extra_conv_channels
                    if len(last_fixed_param) > 0:
                        expid = '{}_FixParam.{}'.format(expid,
                                last_fixed_param.replace('/', '.'))
                        kwargs['last_fixed_param'] = last_fixed_param
                    if data.startswith('office'):
                       assert 'taxonomy_folder' not in kwargs
                       kwargs['taxonomy_folder'] = \
                            './aux_data/taxonomy10k/office/{}'.format(data)
                    expid = expid + suffix
                    kwargs['monitor_train_only'] = monitor_train_only
                    kwargs['expid'] = expid
                    kwargs['net'] = net
                    kwargs['data'] = data
                    all_task.append(kwargs)

    logging.info(pformat(all_task))
    #all_gpus = [[4,5,6,7], [0,1,2,3]]
    all_gpus = [[0,1,2,3], [4,5,6,7]]
    #all_gpus = [[4,5,6,7]]
    #all_gpus = [[0,1]]
    #all_gpus = [[0]]
    #all_gpus = [[0,1,2,3]]
    #all_gpus = [[4, 5, 6, 7]]
    #all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[0]]
    machines = get_machine()
    vigs = machines['vigs']
    all_resource = []
    #all_resource += [(vig2, r) for r in all_gpus]
    #all_resource += [(cluster8, r) for r in all_gpus]
    #all_resource += [(cluster8_2, r) for r in all_gpus]
    all_resource += [(vigs[1], [0,1,2,3])]
    #all_resource += [(vigs[1], [4,5,6,7])]
    #all_resource += [(vigs[1], [0, 1, 2, 3, 4,5,6,7])]
    #all_resource += [(vigs[0], [0, 1, 2, 3, 4,5,6,7])]
    #import ipdb;ipdb.set_trace()
    #all_resource += [(vigs[0], [4,5,6,7])]
    #all_resource += [(vigs[0], [0,1,2,3])]
    #for c in clusters8:
        #for g in [[0,1,2,3,4,5,6,7]]:
            #all_resource += [(c, g)]
    #for c in clusters4:
        #for g in [[0,1,2,3]]:
            #all_resource += [(c, g)]
    #for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
            #all_resource += [(c, g)]
    #for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
            #all_resource += [(c, g)]
    #all_resource += [(vigs[1], [0, 1, 2, 3, 4,5,6,7] * 7)]
    #all_resource += [(vigs[1], [4])]
    #if batch_size_factor == 2:
        #all_resource = []
        #all_resource += [(vigs[0], [0,1,2,3,4,5,6,7])]
        #all_resource += [(vigs[1], [0,1,2,3,4,5,6,7])]
        #all_resource += [(clusters8[0], [0,1,2,3,4,5,6,7])]
    #all_resource += [(clusters8[0], [0,1,2,3])]
    #all_resource += [(clusters8[0], [0,1,2,3,4,5,6,7])]
    #all_resource += [(vigs[1], [0,1,2,3,4,5,6,7])]
    #all_resource += [(cluster8_2, [0,1,2,3])]
    #all_resource += [(cluster8_2, [4,5,6,7])]
    #all_resource += [(cluster4, r) for r in all_gpus]
    #all_resource += [(cluster2_1, [0, 1])]
    #all_resource += [(cluster2_2, [0, 1])]
    logging.info(pformat(all_resource))
    logging.info('#resource: {}'.format(len(all_resource)))
    logging.info('#task: {}'.format(len(all_task)))
    debug = True
    #all_task[-1]['force_evaluate'] = True
    #all_task[-1]['force_predict'] = True
    #debug = False
    #return
    #tsv_file = './data/office_v2.1_with_bb/test.tsv'
    #all_task[0]['force_predict'] = True
    #task = all_task[0]
    #task['expid'] = '{}_bb_nobb'.format(task['expid'])
    ##task['expid'] = '{}_bb_only'.format(task['expid'])
    #task['class_specific_nms'] = False
    #task['yolo_test_thresh'] = 0.5
    #c = CaffeWrapper(**task)
    #c.demo(None)
    #rows = tsv_reader(tsv_file)
    #for row in rows:
        #continue
    #import ipdb;ipdb.set_trace()
    #c.demo('./data/office100_v1_with_bb/train.tsv')
    #c.demo(tsv_file)
    #c.demo('/raid/jianfw/data/office100_crawl/TermList.instagram.pinterest.scrapping.image.tsv')
    #c.demo('/raid/jianfw/work/yuxiao_crop/ring/')
    #c.demo('tmp.png')
    #all_task[0]['force_predict'] = True
    #return
    def batch_run():
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
        #if not monitor_train_only:
            #for t in all_task:
                #t['monitor_train_only'] = True
            #for i, r in enumerate(all_resource):
                #all_resource[i] = (r[0], [-1] * 4)
            #b = BatchProcess(all_resource, all_task, task_processor)
            ##b._availability_check = False
            #b.run()
    if debug:
        idx = -1
        task = all_task[idx]
        task['effective_batch_size'] = 4
        #task['use_pretrained'] = False
        #all_task[idx]['max_iters'] = 1
        #task['expid'] = '{}_debug'.format(expid)
        #all_task[idx]['datas'] = ['voc20', 'crawl_office_v1']
        #all_task[idx]['force_train'] = True
        #all_task[idx]['debug_train'] = True
        #all_task[idx]['debug_detect'] = True
        #all_task[idx]['force_predict'] = True
        #task_processor(({}, [0]), all_task[idx])
        #task['force_evaluate'] = True
        #task['multiscale'] = False
        #task['ovthresh'] = [0.3]
        task_processor(({}, [0]), task)
        #task_processor(all_resource[-1], task)
        #task_processor((vig[1], [0]), task)
        #import ipdb;ipdb.set_trace()
    else:
        batch_run()
        pass

def officev2_1():
    all_task = []
    max_num = 500
    all_task = []
    machine_ips = []
    #for monitor_train_only in [True]:
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    num_extra_convs = 3
    #suffix = '_1'
    suffix = '_2gpu'
    #suffix = '_xx'
    #dataset_ops_template = []
    #dataset_ops = []
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    batch_size_factor = 2
    suffix = '_batchSizeFactor{}'.format(batch_size_factor) \
            if batch_size_factor != 1 else ''
    #suffix = '{}{}'.format(suffix, '_256e')
    suffix = '_withNoBB' 
    #suffix = '' 
    effective_batch_size = 64 * batch_size_factor
    #max_iters=1
    def gen_anchor_bias(num_anchor):
        if num_anchor == 0:
            return None
        if num_anchor == 2:
            return [4, 8, 8, 4]
        result = []
        n = int(np.sqrt(num_anchor))
        assert n * n == num_anchor
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 9
    anchor_bias = gen_anchor_bias(num_anchor)
    anchor_bias = None
    num_anchor = 5
    multibin_wh = False
    multibin_wh_low = 0;
    multibin_wh_high = 13;
    all_multibin_wh_count = [16]
    #all_multibin_wh_count = [16, 32, 48, 64];
    yolo_rescore = True
    yolo_xy_kl_distance = False
    yolo_obj_only = False
    yolo_exp_linear_wh = False
    yolo_obj_nonobj_align_to_iou = False
    yolo_obj_ignore_center_around = False
    yolo_obj_kl_distance = False
    yolo_obj_set1_center_around = False
    yolo_blame = 'xy.wh.obj.cls.nonobj'
    yolo_blame = 'xy.wh.obj.nonobj.cls'
    yolo_blame = ''
    yolo_deconv_to_increase_dim = False
    yolo_deconv_to_increase_dim_adapt_bias = True
    yolo_anchor_aligned_images = 1280000000
    yolo_anchor_aligned_images = 12800
    #yolo_anchor_aligned_images = 0
    yolo_nonobj_extra_power = 0
    yolo_obj_extra_power = 0
    yolo_disable_data_augmentation = False
    yolo_disable_data_augmentation_except_shift = False
    #yolo_anchor_aligned_images = 0
    bn_no_train = False
    yolo_coords_only = False
    if yolo_coords_only:
        yolo_object_scale = 0
        yolo_noobject_scale = 0
        yolo_class_scale = 0
    else:
        yolo_object_scale = 5
        yolo_noobject_scale = 1
        yolo_class_scale = 1
    yolo_avg_replace_max = False
    yolo_per_class_obj = False
    max_iters = '128e'
    data = 'voc20'
    #data = 'brand1048'
    #data = 'office100_v1'
    if data == 'voc20':
        max_iters = 10000
    elif data == 'fridge_clean':
        max_iters = 10000
    #max_iters=10000 / batch_size_factor
    burn_in = '5e'
    burn_in = ''
    burn_in_power = 1

    yolo_full_gpu = True
    yolo_full_gpu = False
    yolo_tree = True
    #yolo_tree = False
    test_tree_cls_specific_th_by_average = 1.2
    test_tree_cls_specific_th_by_average = None
    yolo_angular_loss = True
    yolo_angular_loss = False
    net_input_size_min = 416
    net_input_size_max = 416
    no_bias = 'conf'
    no_bias = ''
    init_from = {'data': 'imagenet', 'net': 'darknet19_448', 'expid': 'A'}
    init_from = {'data': 'office_v2.1', 'net': 'darknet19_448', 
        'expid': 'A_burnIn5e.1_tree_initFrom.imagenet.A'}
    #init_from = {}
    kwargs_template = dict(
            detmodel='yolo',
            max_iters='128e',
            #max_iters='80e',
            #max_iters='256e',
            #max_iters=max_iters,
            #max_iters=11000,
            #yolo_blame=yolo_blame,
            #expid=expid,
            #yolo_jitter=0,
            #yolo_hue=0,
            #yolo_test_fix_xy = True,
            #yolo_test_fix_wh = True,
            #yolo_extract_target_prediction = True,
            #yolo_max_truth=300,
            #yolo_exposure=1,
            #test_on_train = True,
            #yolo_saturation=1,
            #yolo_random_scale_min=1,
            #yolo_random_scale_max=1,
            #expid='A_multibin_wh_0_13_16_no2-wh',
            #expid='baseline_2',
            #snapshot=1000,
            #snapshot=0,
            #target_synset_tree='./aux_data/yolo/9k.tree',
            #target_synset_tree='./data/{}/tree.txt'.format(data),
            #dataset_ops=dataset_ops,
            #effective_batch_size=1,
            #num_anchor=3,
            #num_anchor=num_anchor,
            #force_train=True,
            #force_evaluate=True,
            #debug_detect=True,
            #force_predict=True,
            #extract_features='angular_loss.softmax_loss.o_obj_loss.xy_loss.wh_loss.o_noobj_loss',
            #data_dependent_init=True,
            #restore_snapshot_iter=-1,
            #display=0,
            #region_debug_info=10,
            #display=100,
            #stagelr=stagelr,
            #anchor_bias=anchor_bias,
            #test_input_sizes=[288, 416, 480, 608],
            #test_input_sizes=[416, 608],
            #test_input_sizes=[608, 416],
            #stageiter=[1000, 1000000],
            #stageiter=[100,5000,9000,10000000],
            #stageiter = (np.asarray([5000, 9000, 1000000]) / batch_size_factor).tolist(),
            #stagelr = (np.asarray([0.001,0.0001,0.00001]) * batch_size_factor).tolist(),
            #stagelr=[0.0001,0.001,0.0001,0.0001],
            #burn_in=100,
            #class_specific_nms=False,
            #basemodel='./output/imagenet_darknet19_448_A/snapshot/model_iter_570640.caffemodel',
            #effective_batch_size=effective_batch_size,
            #solver_debug_info=True,
            #yolo_test_maintain_ratio = True,
            ovthresh = [0,0.1,0.2,0.3,0.4,0.5])

    if effective_batch_size != 64:
        kwargs_template['effective_batch_size'] = effective_batch_size

    if yolo_disable_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
        kwargs_template['yolo_fix_offset'] = True
        kwargs_template['yolo_mirror'] = False
    elif yolo_disable_data_augmentation_except_shift:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1

    continue_less_data_augmentation = False
    if continue_less_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
    if batch_size_factor == 2:
        burn_in = '5e'
        burn_in_power = 1

    #adv = [(False, True), (True, False), (True, True)]
    #max_iters = 4
    #for multibin_wh_count in all_multibin_wh_count:
    multibin_wh_count = all_multibin_wh_count[0]
    #for yolo_obj_ignore_center_around, yolo_obj_kl_distance in adv:
    yolo_fixed_target = False
    yolo_obj_nonobj_nopenaltyifsmallthaniou = False
    yolo_obj_cap_center_around = False
    yolo_multibin_xy = False
    yolo_sigmoid_xy = True
    yolo_delta_region3 = False
    yolo_background_class = False
    #yolo_background_class = False
    yolo_use_background_class_to_reduce_obj = 0.4
    #for multibin_xy_count in [32, 16, 8]:
    multibin_xy_count = 4
    #res_loss = True
    res_loss = False
    yolo_use_background_class_to_reduce_obj = 1
    monitor_train_only = False
    #monitor_train_only = True
    #for yolo_use_background_class_to_reduce_obj in [1, 0.8, 0.6, 0.4, 0.2]:
    dataset = TSVDataset(data)
    yolo_low_shot_regularizer = False
    #full_labels = dataset.load_labelmap()
    full_labels = []
    #for low_shot_label in full_labels:
    yolo_random_scale_max = 2
    scale_relative_input = True
    scale_relative_input = False
    nms_type = 'LinearSoft'
    nms_type = 'GaussianSoft'
    nms_type = ''
    gaussian_nms_sigma = 0.5
    scale_constrained_by_one_box_area_min = 0.001
    last_fixed_param = 'dark5e/leaky'
    last_fixed_param = ''
    if test_tree_cls_specific_th_by_average:
        assert yolo_tree
    #for low_shot_label in ['']:
    #for low_shot_label in [full_labels[0]]:
    full_labels.insert(0, '')
    #for low_shot_label in full_labels[:5]:
    low_shot_label = ''
    #for low_shot_label in ['']:
    #for extra_conv_kernel in [[1, 1], [1, 3], [3, 1]]:
    for extra_conv_kernel in [[3, 3, 3]]:
    #for extra_conv_channels in [[1024, 512, 512], [512, 512, 1024], [512, 1024,
        #512]]:
        extra_conv_channels = [1024, 1024, 1024]
    #for extra_conv_channels in [[1024, 1024]]:
    #for extra_conv_channels in [[1024, 1024, 1024]]:
    #for extra_conv_kernel in [[3,3,3], [1, 1, 1], [1, 3, 1], [3, 1, 1], [1, 1, 3], [3,
        #3, 1], [3, 1, 3], [1, 3, 3]]:
    #for extra_conv_kernel in [[1, 3, 1]]:
        if len(low_shot_label) > 0:
            dataset_ops = [{'op':'low_shot', 'labels': 'dog', 'num_train': 1}]
            dataset_ops[0]['labels'] = low_shot_label
            dataset_ops[0]['labels_idx'] = full_labels.index(low_shot_label)
        else:
            dataset_ops = [{'op':'remove'},
                    {'op':'add',
                     'name':'office_v2.1_with_bb',
                     'source':'train',
                     'weight': 3},
                    {'op': 'add',
                     'name': 'office_v2.1_no_bb',
                     'source': 'train',
                     'weight': 1},
                    ]
            #dataset_ops = []
            #for multibin_xy_count in [4]:
        yolo_multibin_xy_low = 0.5 / multibin_xy_count
        #multibin_xy_count = 16
        yolo_multibin_xy_high = 1 - 0.5 / multibin_xy_count
        yolo_multibin_xy_count = multibin_xy_count
        #for yolo_obj_kl_distance in [False]:
        yolo_obj_kl_distance = False
    #for yolo_object_scale in [40, 60]:
        #for monitor_train_only in [False, True]:
        #for monitor_train_only in [True]:
        #for yolo_iou_th_to_use_bkg_cls in [0.1]:
        yolo_iou_th_to_use_bkg_cls = 0.1
        #for last_conv_bias in [True]:
        last_conv_bias = True
        #for rotate_max in [5, 10, 15, 20]:
        rotate_max = 10
        rotate_max = 0
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, True), (True, False)]:
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, False)]:
        incorporate_at_least_one_box, scale_constrained_by_one_box_area = False, False
        #for yolo_angular_loss_weight in [0.1, 1, 10]:
        #for yolo_angular_loss_weight in [1]:
        yolo_angular_loss_weight = 1
        #for data in ['fridge_clean', 'voc20']:
        for data in ['office_v2.1']:
        #for data in ['voc20']:
        #rotate_max = 10
        #rotate_max = 0
            rotate_with_90 = False
        #for yolo_random_scale_min in [0.5, 0.75]:
            yolo_random_scale_min = 0.25
            yolo_random_scale_max = 2
        #for monitor_train_only in [False]:
        #monitor_train_only = True
        #monitor_train_only = True
    #for monitor_train_only in [True]:
    #for monitor_train_only in [False, True]:
            #for add_reorg, num_extra_convs in [(True, [6]), (True, [7])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(False, [3]), (True, [3])]:
            for add_reorg, num_extra_convs in [(True, [3])]:
        #for add_reorg, num_extra_convs in all_extra_option:
        #for weight_decay in [0.0001, 0.00005]:
        #for weight_decay in [0.0005]:
            #weight_decay = 0.0005
            #weight_decay = 0
            #for net in ['darknet19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
            #for net in ['resnet34']:
            #net = 'darknet19'
            #for net in ['resnet101']:
            #for yolo_coord_scale in [1]:
                yolo_coord_scale = 1
                #for data in ['voc20', 'voc2012', 'coco2017', 'imagenet']:
                #for data in ['coco2017']:
                #for weight_decay in [0.001, 0.005, 0.01, 0.05]:
                weight_decay = 0.0005
                #for net in ['resnet34', 'darknet19_448']:
                for net in ['darknet19_448']:
                #for net in ['resnet34']:
                    if len(burn_in) > 0 and data != 'voc20':
                        kwargs_template['stageiter'] = ['60e', '90e', '900e'] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()
                    elif len(burn_in) > 0:
                        kwargs_template['stageiter'] = [5000, 9000, 100000] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()

                    kwargs = copy.deepcopy(kwargs_template)
                    expid = 'A'
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if len(num_extra_convs) == 0 and num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                        kwargs['num_extra_convs'] = num_extra_convs
                    elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3 \
                            or len(num_extra_convs) > 1:
                        expid = expid + '_extraconv{}'.format(
                                '_'.join(map(str, num_extra_convs)))
                        kwargs['num_extra_convs'] = num_extra_convs
                    if multibin_wh:
                        expid = expid + '_multibin_wh_{}_{}_{}'.format(multibin_wh_low,
                                multibin_wh_high, multibin_wh_count)
                        kwargs['multibin_wh'] = multibin_wh
                        kwargs['multibin_wh_low'] = multibin_wh_low
                        kwargs['multibin_wh_high'] = multibin_wh_high
                        kwargs['multibin_wh_count'] = multibin_wh_count
                    if not add_reorg:
                        kwargs['add_reorg'] = add_reorg
                    if num_anchor != 5:
                        expid = '{}_numAnchor{}'.format(expid, num_anchor)
                        assert len(anchor_bias) == 2 * num_anchor
                        kwargs['anchor_bias'] = anchor_bias
                    if not yolo_rescore:
                        expid = '{}_{}'.format(expid, 'norescore')
                        kwargs['yolo_rescore'] = yolo_rescore
                    if yolo_obj_ignore_center_around:
                        expid = '{}_{}'.format(expid, 'ignore')
                        kwargs['yolo_obj_ignore_center_around'] = yolo_obj_ignore_center_around
                    if yolo_obj_kl_distance:
                        expid = '{}_{}'.format(expid, 'objkl')
                        kwargs['yolo_obj_kl_distance'] = yolo_obj_kl_distance
                    if yolo_xy_kl_distance:
                        expid = '{}_xykl'.format(expid)
                        kwargs['yolo_xy_kl_distance'] = yolo_xy_kl_distance
                    if yolo_obj_only:
                        expid = '{}_objonly'.format(expid)
                        kwargs['yolo_obj_only'] = yolo_obj_only
                    if yolo_exp_linear_wh:
                        expid = '{}_explinearwh'.format(expid)
                        kwargs['yolo_exp_linear_wh'] = yolo_exp_linear_wh
                    if yolo_obj_nonobj_align_to_iou:
                        expid = '{}_nonobjtoiou'.format(expid)
                        kwargs['yolo_obj_nonobj_align_to_iou'] = yolo_obj_nonobj_align_to_iou
                    if yolo_obj_set1_center_around:
                        expid = '{}_around1'.format(expid)
                        kwargs['yolo_obj_set1_center_around'] = yolo_obj_set1_center_around
                    if yolo_obj_nonobj_nopenaltyifsmallthaniou:
                        expid = '{}_nolosssmallthaniou'.format(expid)
                        kwargs['yolo_obj_nonobj_nopenaltyifsmallthaniou'] = yolo_obj_nonobj_nopenaltyifsmallthaniou
                    if yolo_obj_cap_center_around:
                        expid = '{}_objcaparound'.format(expid)
                        kwargs['yolo_obj_cap_center_around'] = yolo_obj_cap_center_around
                    if weight_decay != 0.0005:
                        expid = '{}_decay{}'.format(expid, weight_decay)
                        kwargs['weight_decay'] = weight_decay
                    if yolo_fixed_target:
                        expid = '{}_fixedtarget'.format(expid)
                        kwargs['yolo_fixed_target'] = yolo_fixed_target
                    if yolo_deconv_to_increase_dim:
                        expid = '{}_deconvincreasedim'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim'] = True
                    if yolo_coord_scale != 1:
                        expid = '{}_coordscale{}'.format(expid,
                                yolo_coord_scale)
                        kwargs['yolo_coord_scale'] = yolo_coord_scale
                    if not yolo_deconv_to_increase_dim_adapt_bias:
                        expid = '{}_nobiaseadapt'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim_adapt_bias'] = \
                            False
                    if yolo_anchor_aligned_images != 12800:
                        expid = '{}_align{}'.format(expid,
                                yolo_anchor_aligned_images)
                        kwargs['yolo_anchor_aligned_images'] = yolo_anchor_aligned_images
                    if yolo_nonobj_extra_power != 0:
                        expid = '{}_nonobjpower{}'.format(expid,
                                yolo_nonobj_extra_power)
                        kwargs['yolo_nonobj_extra_power'] = yolo_nonobj_extra_power
                    if yolo_obj_extra_power != 0:
                        expid = '{}_objpower{}'.format(expid,
                                yolo_obj_extra_power)
                        kwargs['yolo_obj_extra_power'] = yolo_obj_extra_power
                    if yolo_multibin_xy:
                        kwargs['yolo_multibin_xy'] = yolo_multibin_xy
                        kwargs['yolo_multibin_xy_low'] = yolo_multibin_xy_low
                        kwargs['yolo_multibin_xy_high'] = yolo_multibin_xy_high
                        kwargs['yolo_multibin_xy_count'] = yolo_multibin_xy_count
                        expid = '{}_multibinXY{}'.format(expid,
                                yolo_multibin_xy_count)
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'select_top':
                        expid = '{}_selectTop{}'.format(expid,
                                dataset_ops[0]['num_top'])
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'low_shot':
                        low_shot_labels = dataset_ops[0]['labels']
                        low_shot_num_train = dataset_ops[0]['num_train']
                        expid = '{}_lowShot.{}.{}'.format(expid, 
                                low_shot_labels,
                                low_shot_num_train)
                    if len(dataset_ops) > 0:
                        kwargs['dataset_ops'] = dataset_ops
                    if yolo_disable_data_augmentation:
                        expid = '{}_noAugmentation'.format(expid)
                    if yolo_disable_data_augmentation_except_shift:
                        expid = '{}_noAugExpShift'.format(expid)
                    if bn_no_train:
                        expid = '{}_bnNoTrain'.format(expid)
                        kwargs['bn_no_train'] = bn_no_train
                    if yolo_object_scale != 5:
                        expid = '{}_objscale{}'.format(expid, yolo_object_scale)
                        kwargs['yolo_object_scale'] = yolo_object_scale
                    if yolo_noobject_scale != 1:
                        expid = '{}_noobjScale{}'.format(expid,
                                yolo_noobject_scale)
                        kwargs['yolo_noobject_scale'] = yolo_noobject_scale
                    if yolo_class_scale != 1:
                        expid = '{}_clsScale{}'.format(expid,
                                yolo_class_scale)
                        kwargs['yolo_class_scale'] = yolo_class_scale
                    if yolo_avg_replace_max:
                        expid = '{}_avgReplaceMax'.format(expid)
                        kwargs['yolo_avg_replace_max'] = yolo_avg_replace_max
                    if not yolo_sigmoid_xy:
                        expid = '{}_nosigmoidXY'.format(expid)
                        kwargs['yolo_sigmoid_xy'] = yolo_sigmoid_xy
                    if yolo_delta_region3:
                        expid = '{}_deltaRegion3'.format(expid)
                        kwargs['yolo_delta_region3'] = yolo_delta_region3
                    if yolo_background_class:
                        expid = '{}_bkgCls{}'.format(expid,
                                yolo_use_background_class_to_reduce_obj)
                        kwargs['yolo_background_class'] = True
                        kwargs['yolo_use_background_class_to_reduce_obj'] = yolo_use_background_class_to_reduce_obj
                        if yolo_iou_th_to_use_bkg_cls != 1:
                            expid = '{}_iouTh{}'.format(expid,
                                    yolo_iou_th_to_use_bkg_cls)
                            kwargs['yolo_iou_th_to_use_bkg_cls'] = yolo_iou_th_to_use_bkg_cls
                    if res_loss:
                        expid = '{}_resLoss'.format(expid)
                        kwargs['res_loss'] = res_loss
                        kwargs['skip_genprototxt'] = True
                    if yolo_per_class_obj:
                        expid = '{}_perClassObj'.format(expid)
                        kwargs['yolo_per_class_obj'] = yolo_per_class_obj
                    if not last_conv_bias:
                        expid = '{}_noBiasLastConv'.format(expid)
                        kwargs['yolo_last_conv_bias'] = last_conv_bias
                    if yolo_low_shot_regularizer:
                        expid = '{}_lowShotEqualNorm'.format(expid)
                        kwargs['yolo_low_shot_regularizer'] = True
                    if yolo_full_gpu:
                        expid = '{}_fullGpu'.format(expid)
                        kwargs['yolo_full_gpu'] = yolo_full_gpu
                    if burn_in != '':
                        expid = '{}_burnIn{}.{}'.format(expid, burn_in,
                                burn_in_power)
                        kwargs['burn_in'] = burn_in
                        kwargs['burn_in_power'] = burn_in_power
                    if rotate_max != 0:
                        kwargs['rotate_max'] = rotate_max
                        expid = '{}_rotate{}'.format(expid, rotate_max)
                    if rotate_with_90:
                        expid = '{}_with90'.format(expid)
                        kwargs['rotate_with_90'] = True
                    if yolo_random_scale_min != 0.25:
                        expid = '{}_randomScaleMin{}'.format(expid, 
                                yolo_random_scale_min)
                        kwargs['yolo_random_scale_min'] = yolo_random_scale_min
                    if yolo_random_scale_max != 2:
                        expid = '{}_randomScaleMax{}'.format(expid,
                                yolo_random_scale_max)
                        kwargs['yolo_random_scale_max'] = yolo_random_scale_max
                    if scale_relative_input:
                        expid = '{}_RelativeScale2'.format(expid)
                        kwargs['scale_relative_input'] = scale_relative_input
                    if nms_type != 'Standard' and nms_type != '':
                        if nms_type == 'LinearSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.LinearSoft 
                        if nms_type == 'GaussianSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.GaussianSoft
                        if gaussian_nms_sigma != 0.5:
                            kwargs['gaussian_nms_sigma'] = 0.5
                    if incorporate_at_least_one_box:
                        expid = '{}_atLeastOneBB'.format(expid)
                        kwargs['incorporate_at_least_one_box'] = incorporate_at_least_one_box
                    if scale_constrained_by_one_box_area:
                        expid = '{}_scaleConstrainedByOne'.format(expid)
                        kwargs['scale_constrained_by_one_box_area'] = scale_constrained_by_one_box_area
                        if scale_constrained_by_one_box_area_min != 0.001:
                            expid = '{}_Min{}'.format(expid, scale_constrained_by_one_box_area_min)
                            kwargs['scale_constrained_by_one_box_area_min'] = scale_constrained_by_one_box_area_min
                    if yolo_tree:
                        expid = '{}_tree'.format(expid)
                        kwargs['yolo_tree'] = yolo_tree
                        if test_tree_cls_specific_th_by_average is not None:
                            kwargs['test_tree_cls_specific_th_by_average'] = test_tree_cls_specific_th_by_average
                    if len(init_from) > 0:
                        assert net == init_from['net']
                        c = CaffeWrapper(data=init_from['data'], 
                                net=init_from['net'],
                                expid=init_from['expid'])
                        kwargs['basemodel'] = c.best_model().model_param
                        exid = '{}_initFrom{}'.format(expid,
                                hash(kwargs['basemodel']) % 10000)
                    if yolo_angular_loss:
                        expid = '{}_AngularRegulizer'.format(expid)
                        kwargs['yolo_angular_loss'] = True
                        if yolo_angular_loss_weight != 1:
                            expid = '{}Weight{}'.format(expid,
                                    yolo_angular_loss_weight)
                            kwargs['yolo_angular_loss_weight'] = yolo_angular_loss_weight
                    if len(no_bias) > 0:
                        expid = '{}_noBias{}'.format(expid, no_bias)
                        kwargs['no_bias'] = no_bias
                    if net_input_size_min != 416:
                        expid = '{}_InMin{}'.format(expid, net_input_size_min)
                        kwargs['net_input_size_min'] = net_input_size_min
                    if net_input_size_max != 416:
                        expid = '{}_InMax{}'.format(expid, net_input_size_max)
                        kwargs['net_input_size_max'] = net_input_size_max
                    if any(k != 3 for k in extra_conv_kernel):
                        expid = '{}_extraConvKernel.{}'.format(expid, '.'.join(
                            map(str, extra_conv_kernel)))
                        kwargs['extra_conv_kernel'] = extra_conv_kernel
                    if any(c != 1024 for c in extra_conv_channels):
                        expid = '{}_extraChannels.{}'.format(expid,
                                '.'.join(map(str, extra_conv_channels)))
                        kwargs['extra_conv_channels'] = extra_conv_channels
                    if len(last_fixed_param) > 0:
                        expid = '{}_FixParam.{}'.format(expid,
                                last_fixed_param.replace('/', '.'))
                        kwargs['last_fixed_param'] = last_fixed_param
                    expid = expid + suffix
                    kwargs['monitor_train_only'] = monitor_train_only
                    kwargs['expid'] = expid
                    kwargs['net'] = net
                    kwargs['data'] = data
                    all_task.append(kwargs)

    logging.info(pformat(all_task))
    #all_gpus = [[4,5,6,7], [0,1,2,3]]
    all_gpus = [[0,1,2,3], [4,5,6,7]]
    #all_gpus = [[4,5,6,7]]
    #all_gpus = [[0,1]]
    #all_gpus = [[0]]
    #all_gpus = [[0,1,2,3]]
    #all_gpus = [[4, 5, 6, 7]]
    #all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[0]]
    vigs, clusters8 = get_machine()
    all_resource = []
    #all_resource += [(vig2, r) for r in all_gpus]
    #all_resource += [(cluster8, r) for r in all_gpus]
    #all_resource += [(cluster8_2, r) for r in all_gpus]
    #all_resource += [(vigs[1], [0,1,2,3])]
    #all_resource += [(vigs[1], [4,5,6,7])]
    all_resource += [(vigs[1], [0, 1, 2, 3, 4,5,6,7])]
    #all_resource += [(vigs[0], [4,5,6,7])]
    #all_resource += [(vigs[0], [0,1,2,3])]
    #for c in clusters8:
        #for g in [[0,1,2,3,4,5,6,7]]:
            #all_resource += [(c, g)]
    #for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
            #all_resource += [(c, g)]
    #for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
            #all_resource += [(c, g)]
    #all_resource += [(vigs[1], [0, 1, 2, 3, 4,5,6,7] * 7)]
    #all_resource += [(vigs[1], [4])]
    #if batch_size_factor == 2:
        #all_resource = []
        #all_resource += [(vigs[0], [0,1,2,3,4,5,6,7])]
        #all_resource += [(vigs[1], [0,1,2,3,4,5,6,7])]
        #all_resource += [(clusters8[0], [0,1,2,3,4,5,6,7])]
    #all_resource += [(clusters8[0], [0,1,2,3])]
    #all_resource += [(clusters8[0], [0,1,2,3,4,5,6,7])]
    #all_resource += [(vigs[1], [0,1,2,3,4,5,6,7])]
    #all_resource += [(cluster8_2, [0,1,2,3])]
    #all_resource += [(cluster8_2, [4,5,6,7])]
    #all_resource += [(cluster4, r) for r in all_gpus]
    #all_resource += [(cluster2_1, [0, 1])]
    #all_resource += [(cluster2_2, [0, 1])]
    logging.info(pformat(all_resource))
    logging.info('#resource: {}'.format(len(all_resource)))
    logging.info('#task: {}'.format(len(all_task)))
    #debug = True
    #all_task[-1]['force_predict'] = True
    debug = False
    #return
    tsv_file = './data/office_v2.1_with_bb/test.tsv'
    #all_task[0]['force_predict'] = True
    c = CaffeWrapper(**all_task[0])
    rows = tsv_reader(tsv_file)
    #for row in rows:
        #continue
    #import ipdb;ipdb.set_trace()
    #c.demo('./data/office100_v1_with_bb/train.tsv')
    #c.demo(tsv_file)
    #c.demo('/raid/jianfw/data/office100_crawl/TermList.instagram.pinterest.scrapping.image.tsv')
    #c.demo('/raid/jianfw/work/yuxiao_crop/ring/')
    #c.demo('tmp.png')
    c.demo(None)
    #all_task[0]['force_predict'] = True
    return
    def batch_run():
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
        if not monitor_train_only:
            for t in all_task:
                t['monitor_train_only'] = True
            for i, r in enumerate(all_resource):
                all_resource[i] = (r[0], [-1] * 4)
            b = BatchProcess(all_resource, all_task, task_processor)
            #b._availability_check = False
            b.run()
    if debug:
        idx = -1
        all_task[idx]['effective_batch_size'] = 4
        #all_task[idx]['max_iters'] = 1
        all_task[idx]['force_train'] = True
        all_task[idx]['debug_train'] = True
        #all_task[idx]['debug_detect'] = True
        #all_task[idx]['force_predict'] = True
        #task_processor(({}, [0]), all_task[idx])
        for task in all_task:
            #task['force_evaluate'] = True
            task_processor(({}, [4]), task)
            #import ipdb;ipdb.set_trace()
    else:
        batch_run()
        pass

def smaller_network_input_size():
    init_logging()
    #check_net()
    #return
    data = 'imagenet'
    data = 'voc20'
    #data = 'coco'
    dataset = TSVDataset(data)
    max_num = 500
    all_task = []
    #net = 'resnet18'
    net = 'darknet19'
    #for monitor_train_only in [False, True]:
    #for monitor_train_only in [False]:
    all_gpus = [[4,5,6,7], [0,1,2,3]]
    all_gpus = [[4,5,6,7], [-1] * 8]
    all_gpus = [[-1] * 8]
    vig1 = {'username':'REDMOND.jianfw',
            'ip':'vig-gpu01.guest.corp.microsoft.com'}
    dl1 = {'username': 'jianfw',
            'ip':'10.196.44.72',
            '-p': 30144,
            'data': '/work/data/qd_data_cluster',
            'output': '/work/work/qd_output'}
    dl8 = {'username': 'jianfw',
            'ip':'10.196.44.185',
            '-p': 30824,
            'data': '/work/data/qd_data_cluster',
            'output': '/work/work/qd_output'}
    all_resource = []
    #all_resource += [(dl1, r) for r in all_gpus]
    #all_resource += [(dl8, r) for r in all_gpus]
    #all_resource += [(vig1, r) for r in all_gpus]
    all_resource += [({}, r) for r in all_gpus]
    machine_ips = []
    #for monitor_train_only in [True]:
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    add_reorg = True
    num_extra_convs = 3
    #all_extra_option = [(False, 0), (True, 3), (False, 1), (False, 2), (False, 3)]
    #all_extra_option = [(False, 1), (False, 2)]
    #all_extra_option = [(True, 3)]
    all_extra_option = [(True, 3)]
    all_net = ['darknet19', 'resnet18', 'resnet34']
    all_net = ['darknet19']
    suffix = ''
    #suffix = '_2'
    #suffix = '_xx'
    # operation on the training data set
    #dataset_ops = [{'op':'remove'},
            #{'op':'add',
             #'name':'coco',
             #'source':'trainval',
             #'weight': 1}]
    dataset_ops = []
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    effective_batch_size=64
    stagelr = np.asarray([0.0001,0.001,0.0001,0.00001])
    stagelr = np.asarray([0.0001,0.001,0.0001,0.00001])
    stagelr = stagelr.tolist()
    max_iters='128e'
    #max_iters=10000
    def gen_anchor_bias(n):
        if n == 0:
            return None
        if n == 2:
            return [4, 8, 8, 4]
        result = []
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 0
    anchor_bias = gen_anchor_bias(num_anchor)
    yolo_rescore = True
    #max_iters = 4
    add_image_rule = None
    #for monitor_train_only in [False, True]:
    for monitor_train_only in [True]:
        for add_reorg, num_extra_convs in all_extra_option:
            for net in all_net:
                #for network_input_size in [352, 288]:
                for network_input_size in [416]:
                    expid = 'A'
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                    elif num_extra_convs != 3:
                        expid = expid + '_extraconv{}'.format(num_extra_convs)
                    if anchor_bias:
                        expid = '{}_anchorsplit{}'.format(expid,
                                '_'.join(map(str, anchor_bias)))
                    if network_input_size != 416:
                        expid = '{}_netinputsize{}'.format(expid, network_input_size)
                    if not yolo_rescore:
                        expid = '{}_{}'.format(expid, 'norescore')
                    expid = expid + suffix
                    ovthresh = [0,0.1,0.2,0.3,0.4,0.5]
                    logging.info('expid: ' + expid)
                    kwargs = dict(data=data,
                            net=net,
                            detmodel='yolo',
                            max_iters=max_iters,
                            expid=expid,
                            snapshot=500,
                            #target_synset_tree='./aux_data/yolo/9k.tree',
                            network_input_size=network_input_size,
                            dataset_ops=dataset_ops,
                            effective_batch_size=effective_batch_size,
                            num_extra_convs=num_extra_convs,
                            #num_anchor=3,
                            num_anchor=5,
                            #force_train=True,
                            #force_evaluate=True,
                            #restore_snapshot_iter=5000,
                            yolo_rescore=yolo_rescore,
                            #display=10,
                            display=100,
                            stagelr=stagelr,
                            anchor_bias=anchor_bias,
                            add_reorg=add_reorg,
                            #stageiter=[1000, 1000000],
                            #effective_batch_size=16,
                            #solver_debug_info=True,
                            monitor_train_only=monitor_train_only,
                            ovthresh=ovthresh)
                    all_task.append(kwargs)

    logging.info(pformat(all_task))
    logging.info(pformat(all_resource))
    #task_processor(all_resource[0], all_task[0])
    b = BatchProcess(all_resource, all_task, task_processor)
    b.run()

def yolo9000():
    init_logging()
    #check_net()
    #return
    data = 'imagenet'
    #data = 'coco'
    dataset = TSVDataset(data)
    all_task = []
    max_num = 500
    all_task = []
    add_image_rule = 'coco:anyuseful:withbb:500'
    add_image_rule = 'coco:anyuseful:nobb:500'
    add_image_rule = None
    #net = 'resnet18'
    net = 'darknet19'
    #for monitor_train_only in [False, True]:
    #for monitor_train_only in [False]:
    syned_ips = []
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    add_reorg = True
    num_extra_convs = 3
    all_extra_option = [(False, [0]), (True, [3]), (False, [3])]
    #all_extra_option = [(False, 1), (False, 2)]
    all_extra_option = [(False, [0])]
    all_net = ['darknet19', 'resnet18', 'resnet34']
    all_net = ['darknet19']
    #suffix = '_1'
    suffix = '_orig'
    suffix = ''
    # operation on the training data set
    dataset_ops = [{'op':'remove'},
            {'op':'add',
             'name':'coco',
             'source':'trainval',
             'weight': 1},
            {'op': 'add',
                'path': '/raid/jianfw/work/imagenet22k_9ktree',
                'name': 'imagenet22k_9k',
                'source': 'train',
                'weight': 3}]
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    factor = 3
    effective_batch_size=64 * factor
    max_iters = 500200
    stagelr = [0.0001, 0.00001]
    stageiter = [400000, 10000000]
    #max_iters = 4
    for monitor_train_only in [False]:
    #for monitor_train_only in [True]:
        for add_reorg, num_extra_convs in all_extra_option:
            for net in all_net:
                for add_image_rule in [None]:
                    expid = encode_expid('A',
                            (None, add_image_rule),
                            )
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if len(num_extra_convs) == 1 and num_extra_convs[0] == 0:
                        expid = expid + '_noextraconv'
                    elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3:
                        expid = expid + '_extraconv{}'.format(num_extra_convs)
                    expid = expid + suffix
                    ovthresh = [0,0.1,0.2,0.3,0.4,0.5]
                    logging.info('expid: ' + expid)
                    kwargs = dict(data=data,
                            net=net,
                            detmodel='yolo',
                            max_iters=max_iters,
                            expid=expid,
                            snapshot=500,
                            target_synset_tree='./aux_data/yolo/9k.tree',
                            dataset_ops=dataset_ops,
                            effective_batch_size=effective_batch_size,
                            add_image_rule=add_image_rule,
                            num_extra_convs=num_extra_convs,
                            test_data='imagenet22k_9k',
                            test_on_train=True,
                            num_anchor=3,
                            burn_in=1000,
                            #extract_features='label.conf_debug',
                            #force_train=True,
                            #force_evaluate=True,
                            #restore_snapshot_iter=1000,
                            display=10,
                            stagelr=stagelr,
                            stageiter=stageiter,
                            add_reorg=add_reorg,
                            predict_evaluate_loss_per_cls=True,
                            #stageiter=[1000, 1000000],
                            #effective_batch_size=16,
                            #solver_debug_info=True,
                            monitor_train_only=monitor_train_only,
                            ovthresh=ovthresh)
                    all_task.append(kwargs)

    all_gpus = [[4,5,6,7], [0,1,2,3]]
    #all_gpus = [[0,1,2,3,4,5,6,7]]
    vigs = get_machine()
    all_resource = []
    #all_resource += [(dl1, r) for r in all_gpus]
    all_resource += [(vigs[1], r) for r in all_gpus]
    #all_resource += [({}, r) for r in all_gpus]
    assert len(all_task) == 1
    logging.info(pformat(all_task))
    logging.info(pformat(all_resource))

    debug = True
    #debug = False
    #all_task[0]['force_predict'] = True
    #all_task[0]['force_evaluate'] = True
    if debug:
        #assert len(all_task) == 1
        #all_task[0]['effective_batch_size'] = 1
        #all_task[0]['max_iters'] = 10000
        #all_task[0]['max_iters'] = 1 
        #all_task[0]['force_train'] = True
        #all_task[0]['debug_train'] = True
        all_task[0]['debug_detect'] = True
        all_task[0]['force_predict'] = True
        task_processor(({}, [0]), all_task[0])
    #for task in all_task:
        #task_processor(({}, [0]), task)

    #b = BatchProcess(all_resource, all_task, task_processor)
    ##b._availability_check = False
    #b.run()

def yolo9000_coco50K():
    init_logging()
    #check_net()
    #return
    data = 'imagenet'
    #data = 'coco'
    dataset = TSVDataset(data)
    all_task = []
    max_num = 500
    all_task = []
    add_image_rule = 'coco:anyuseful:withbb:500'
    add_image_rule = 'coco:anyuseful:nobb:500'
    add_image_rule = None
    #net = 'resnet18'
    net = 'darknet19'
    #for monitor_train_only in [False, True]:
    #for monitor_train_only in [False]:
    all_gpus = [[4,5,6,7], [0,1,2,3]]
    all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[0,1,2,3]]
    #all_gpus = [[-1] * 16]
    vig1 = {'username':'REDMOND.jianfw',
            'ip':'vig-gpu01.guest.corp.microsoft.com'}
    dl8 = {'username': 'jianfw',
            'ip':'10.196.44.185',
            '-p': 30824,
            'data': '/work/data/qd_data_cluster',
            'output': '/work/work/qd_output'}
    all_resource = []
    #all_resource += [(vig1, r) for r in all_gpus]
    #all_resource += [(dl1, r) for r in all_gpus]
    all_resource += [(dl8, r) for r in all_gpus]
    #all_resource += [({}, r) for r in all_gpus]
    syned_ips = []
    remove_bb_label = None
    remove_bb_label_prob = None
    remove_image_rule = 'all'
    remove_image_rule = None
    remove_image_prob = None
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    add_reorg = True
    num_extra_convs = 3
    all_extra_option = [(False, 0), (True, 3), (False, 3)]
    #all_extra_option = [(False, 1), (False, 2)]
    all_extra_option = [(False, 0)]
    all_net = ['darknet19', 'resnet18', 'resnet34']
    all_net = ['darknet19']
    #suffix = '_1'
    suffix = ''
    # operation on the training data set
    dataset_ops = [{'op':'remove'},
            {'op':'add',
             'name':'coco',
             'source':'trainval',
             'weight': 1}]
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    factor = 1
    effective_batch_size=64 * factor
    max_iters=50000
    stagelr = [0.0001, 0.00001]
    stageiter = [40000, 10000000]
    test_data = 'imagenet'
    #max_iters = 4
    for monitor_train_only in [False]:
    #for monitor_train_only in [True]:
        for add_reorg, num_extra_convs in all_extra_option:
            for net in all_net:
                for add_image_rule in [None]:
                    expid = encode_expid('A',
                            ('ri', remove_image_rule),
                            ('', remove_image_prob),
                            (None, add_image_rule),
                            ('rb', remove_bb_label),
                            (None, remove_bb_label_prob),
                            )
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                    elif num_extra_convs != 3:
                        expid = expid + '_extraconv{}'.format(num_extra_convs)
                    if not remove_image_rule and not add_image_rule and \
                            not remove_bb_label and add_reorg \
                            and num_extra_convs == 3:
                        expid = 'baseline'
                    expid = expid + suffix
                    ovthresh = [0,0.1,0.2,0.3,0.4,0.5]
                    logging.info('expid: ' + expid)
                    kwargs = dict(data=data,
                            test_data=test_data,
                            net=net,
                            detmodel='yolo',
                            max_iters=max_iters,
                            expid=expid,
                            snapshot=10000,
                            target_synset_tree='./aux_data/yolo/9k.tree',
                            dataset_ops=dataset_ops,
                            effective_batch_size=effective_batch_size,
                            num_extra_convs=num_extra_convs,
                            num_anchor=3,
                            #force_train=True,
                            #debug_detect=False,
                            #force_evaluate=False,
                            #force_predict=False,
                            #restore_snapshot_iter=1000,
                            display=1,
                            stagelr=stagelr,
                            stageiter=stageiter,
                            add_reorg=add_reorg,
                            #yolo_test_map='./aux_data/yolo/imagenet200_to_9k.map',
                            #stageiter=[1000, 1000000],
                            #effective_batch_size=16,
                            #solver_debug_info=True,
                            monitor_train_only=monitor_train_only,
                            ovthresh=ovthresh)
                    all_task.append(kwargs)

    assert len(all_task) == 1
    logging.info(pformat(all_resource))
    logging.info(pformat(all_task))

    #task_processor(all_resource[0], all_task[0])
    #b = BatchProcess(all_resource, all_task, task_processor)
    #b.run()

    logging.info('finished the first stage')

    dataset_ops = [{'op':'remove'},
            {'op':'add',
             'name':'coco',
             'source':'trainval',
             'weight': 1},
            {'op': 'add',
                #'path': '/raid/jianfw/work/imagenet22k_9ktree',
                'path': '/work/data/imagenet22k_9ktree',
                'name': 'imagenet22k_9k',
                'source': 'train',
                'weight': 3}]

    max_iters=500200
    stagelr = [0.0001, 0.00001]
    stageiter = [400000, 10000000]
    kwargs['dataset_ops'] = dataset_ops
    kwargs['max_iters'] = max_iters
    kwargs['stagelr'] = stagelr
    kwargs['stageiter'] = stageiter
    kwargs['restore_snapshot_iter'] = 50000

    kwargs['effective_batch_size'] =64 * 3
    all_task = [kwargs]

    logging.info(pformat(all_resource))
    logging.info(pformat(all_task))

    #task_processor(all_resource[0], all_task[0])
    b = BatchProcess(all_resource, all_task, task_processor)
    b.run()

def remove_bb_train_test2():
    init_logging()
    #check_net()
    #return
    data = 'voc20'
    #data = 'coco'
    dataset = TSVDataset(data)
    all_task = []
    max_num = 500
    all_task = []
    all_resource = []
    add_image_rule = 'coco:anyuseful:withbb:500'
    add_image_rule = 'coco:anyuseful:nobb:500'
    add_image_rule = None
    #net = 'resnet18'
    net = 'darknet19'
    #for monitor_train_only in [False, True]:
    #for monitor_train_only in [False]:
    all_gpus = ['4,5,6,7', '0,1,2,3']
    all_gpus = map(str, range(8))
    #all_gpus = ['4,5,6,7']
    vig1 = {'username':'REDMOND.jianfw',
            'ip':'vig-gpu01.guest.corp.microsoft.com'}
    #all_resource = [(vig1, r) for r in all_gpus]
    all_resource += [({}, r) for r in all_gpus]
    #sync(ssh_info=vig1)
    #for monitor_train_only in [True]:
    remove_bb_label = None
    remove_bb_label_prob = None
    remove_image_rule = 'all'
    remove_image_rule = None
    remove_image_prob = None
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    add_reorg = False
    num_extra_convs = 3
    all_extra_option = [(False, 0), (False, 1), (False, 2), (True, 3), (False, 3)]
    #all_extra_option = [(False, 1), (False, 2)]
    #all_extra_option = [(False, 0)]
    all_net = ['darknet19', 'resnet18', 'resnet34']
    #all_net = ['darknet19']
    #suffix = '_1'
    suffix = ''
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    for add_reorg, num_extra_convs in all_extra_option:
    #for monitor_train_only in [False, True]:
        #for remove_bb_label_prob in [0.9, 0.5]:
        #for remove_bb_label_prob in [None]:
        _test_time = {}
        for monitor_train_only in [True]:
        #for remove_image_prob in [0.9, 0.5]:
        #for remove_image_prob in [None]:
        #for remove_bb_label_prob in [0.9, 0.5]:
            #for net in ['resnet34']:
            for net in all_net:
            #for add_image_rule in [None, 'coco:anyuseful:nobb:500',
                    #'coco:anyuseful:withbb:500']:
                for add_image_rule in [None]:
                    expid = encode_expid('A',
                            ('ri', remove_image_rule),
                            ('', remove_image_prob),
                            (None, add_image_rule),
                            ('rb', remove_bb_label),
                            (None, remove_bb_label_prob),
                            )
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if num_extra_convs is list:
                        expid = '{}_extraconv{}'.format(expid, '_'.join(num_extra_convs))
                    else:
                        if num_extra_convs == 0:
                            expid = expid + '_noextraconv'
                        elif num_extra_convs != 3:
                            expid = expid + '_extraconv{}'.format(num_extra_convs)
                    if not remove_image_rule and not add_image_rule and \
                            not remove_bb_label and add_reorg \
                            and num_extra_convs == 3:
                        expid = 'baseline'
                    expid = expid + suffix
                    ovthresh = [0,0.1,0.2,0.3,0.4,0.5]
                    logging.info('expid: ' + expid)
                    kwargs = dict(data=data,
                            net=net,
                            detmodel='yolo',
                            max_iters=10000,
                            expid=expid,
                            snapshot=500,
                            remove_bb_label=remove_bb_label,
                            remove_bb_label_prob=remove_bb_label_prob,
                            remove_image=remove_image_rule,
                            remove_image_prob=remove_image_prob,
                            add_image_rule=add_image_rule,
                            num_extra_convs=num_extra_convs,
                            gpus='0',
                            #force_train=True,
                            #force_evaluate=True,
                            #restore_snapshot_iter=5000,
                            display=100,
                            #stagelr=[0.0001, 0.0001],
                            add_reorg=add_reorg,
                            #stageiter=[1000, 1000000],
                            #effective_batch_size=16,
                            #solver_debug_info=True,
                            monitor_train_only=monitor_train_only,
                            ovthresh=ovthresh)

                    c = CaffeWrapper(**kwargs)
                    if 'cpu' not in _test_time:
                        _test_time['cpu'] = {}
                    if 'gpu' not in _test_time:
                        _test_time['gpu'] = {}
                    _test_time['cpu'][net] = '{:.3f}'.format(c.cpu_test_time())
                    _test_time['gpu'][net] = '{:.3f}'.format(c.gpu_test_time())
                    #_num_param[num_param_key][net] = '{:,}'.format(c.param_num())
                    #m, s, iters = c.training_time()
                    #_training_time[training_time_key][net] = '{:.2f}$\\pm${:.2f}'.format(m, s)
                    #perf = c.best_model_perf()
                    #_report[net] = {}
                    #for th in ovthresh:
                        #_report[net][th] = \
                                #'{:.1f}'.format(perf['overall'][str(th)]['map']*100)

                    all_task.append(kwargs)
        r = print_m_table(_test_time, [['cpu', 'gpu']], [all_net], expid)
        write_to_file(r,
                '/home/jianfw/code/experiments/yolo/testing_time_{}.tex'.format(expid))
        r = print_csv_table(_test_time, ['cpu', 'gpu'], all_net)
        write_to_file(r,
                '/home/jianfw/code/experiments/yolo/testing_time_{}.csv'.format(expid))

    #r = print_m_table(_training_time, [[training_time_key]], [all_net],
            #'Training time cost per {} iterations'.format(iters))
    #write_to_file(r, '/home/jianfw/code/experiments/yolo/training_time_{}.tex'.format(expid))
    #r = print_csv_table(_training_time, [training_time_key], all_net)
    #write_to_file(r,
            #'/home/jianfw/code/experiments/yolo/training_time_{}.csv'.format(expid))

    #r = print_m_table(_report, [all_net], [ovthresh],
            #'MAP for Yolo detection (second run)')
    #write_to_file(r,
            #'/home/jianfw/code/experiments/yolo/map_{}.tex'.format(expid))
    #r = print_csv_table(_report, all_net, ovthresh)
    #write_to_file(r,
            #'/home/jianfw/code/experiments/yolo/map_{}.csv'.format(expid))

    #r = print_m_table(_num_param, [[num_param_key]], [all_net])
    #write_to_file(r,
            #'/home/jianfw/code/experiments/yolo/num_param_{}.tex'.format(expid))
    #r = print_csv_table(_num_param, [num_param_key], all_net)
    #write_to_file(r,
            #'/home/jianfw/code/experiments/yolo/num_param_{}.csv'.format(expid))

    #task_processor(all_resource[0], all_task[1])
    #b = BatchProcess(all_resource, all_task, task_processor)
    #b.run()

def re_run():
    data = 'office_v2.12'
    net = 'darknet19_448'
    expid = 'A_noreorg_burnIn5e.1_tree_initFrom.imagenet.A_bb_nobb'

    kwargs = load_from_yaml_file(op.join('output', '{}_{}_{}'.format(data,
        net, expid), 'parameters.yaml'))
    kwargs['data'] = data
    kwargs['net'] = net

    all_task = []
    all_task.append(kwargs)

    vigs, clusters8, clusters4 = get_machine()
    all_resource = []
    all_resource += [(vigs[1], [0,1,2,3])]
    
    kwargs['force_evaluate'] = True
    task_processor(({}, [0]), kwargs, func=yolotrain)

    #b = BatchProcess(all_resource, all_task, task_processor)
    #b.run()


def plot_remove_bb_result():
    label_removed = 'dog'
    eval_map = 0.5
    #label_removed = 'aeroplane'
    cache_file = 'cache_result_{}_{}.json'.format(label_removed, eval_map)
    expids = {'baseline':'baseline',
            'remove_{}10'.format(label_removed):'box10',
            'remove_{}_image10'.format(label_removed):'image10',
            'remove_{}25'.format(label_removed):'box25',
            'remove_{}_image25'.format(label_removed):'image25',
            'remove_{}50'.format(label_removed):'box50',
            'remove_{}_image50'.format(label_removed):'image50',
            'remove_{}75'.format(label_removed):'box75',
            'remove_{}_image75'.format(label_removed):'image75',
            'remove_{}80'.format(label_removed):'box80',
            'remove_{}_image80'.format(label_removed):'image80',
            'remove_{}90'.format(label_removed):'box90',
            'remove_{}_image90'.format(label_removed):'image90',
            'remove_{}'.format(label_removed):'box100',
            'remove_{}100'.format(label_removed):'box100'}

    #table_rows = ['baseline', 'box10', 'image10', 'box25', 'image25',
            #'box50', 'image50', 'box75', 'image75', 'box80', 'image80',
            #'box90', 'image90',
            #'box100']

    table_rows = ['baseline', 'box10', 'box25', 'box50', 'box75', 'box80',
            'box90', 'box100']

    #use_cache = False
    use_cache = False
    if os.path.isfile(cache_file) and use_cache:
        with open(cache_file, 'r') as fp:
            result = json.load(fp)
    else:
        kwargs = {}
        kwargs['data'] = 'voc20'
        kwargs['net'] = 'darknet19'
        kwargs['max_iters'] = 10000
        kwargs['expid'] = 'baseline'
        kwargs['gpus'] = '0,1,2,3,4,5,6,7'
        kwargs['snapshot'] = 500
        kwargs['monitor_train_only'] = True
        result = {}
        for expid in expids:
            kwargs['expid'] = expid
            c = CaffeWrapper(**kwargs)
            try:
                perf = c.best_model_perf()['overall'][eval_map]
            except IOError:
                continue
            result[expid] = perf
        with open(cache_file, 'w') as fp:
            json.dump(result, fp)

    labels = None
    r = {}
    for expid in expids:
        x = expids[expid]
        if expid not in result:
            continue
        class_ap = result[expid]
        if labels == None:
            labels = class_ap.keys()
        if len(r) == 0:
            for l in labels:
                r[l] = {}
        for label in labels:
            ap = class_ap[label]
            if label not in r:
                r[label][expids[expid]] = ([], [])
            r[label][expids[expid]] = ap

    labels.remove(label_removed)
    labels = [label_removed] + labels
    x = print_table(r, labels, table_rows)
    #r = r_box
    #r = r_image
    #for label in labels:
        #xs, ys = r[label]
        #xs = np.asarray(xs)
        #ys = np.asarray(ys)
        #idx = np.argsort(xs)
        #xs = xs[idx]
        #ys = ys[idx]
        #plt.plot(xs, ys, '-o')
    #plt.grid()
    #plt.xlabel('Prob. of removing the bouding box')
    #plt.ylabel('AP')
    #plt.legend(labels)
    #plt.show()


def remove_bb_train_test():
    data = 'voc20'
    dataset = TSVDataset('voc20')
    all_labels = dataset.load_labelmap()

    #all_resource = ['4,5,6,7']
    all_resource = ['0,1,2,3', '4,5,6,7']
    all_task = []
    add_coco_image = True
    max_num = 500
    is_report = True

    report = {}
    #for prob in [1, 0.1, 0.25, 0.5, 0.75, 0.9]:
    for prob in [1, 0.9]:
        for label in all_labels:
                monitor_train_only = False
                if prob == 0:
                    expid = 'baseline'
                else:
                    expid = 'remove_{}{}'.format(label, int(prob * 100))
                    if add_coco_image:
                        expid = '{}_addcoco{}'.format(expid, max_num)
                if add_coco_image:
                    add_image_rule = 'coco:{}:{}'.format(label, max_num)
                else:
                    add_image_rule = None
                kwargs = dict(data=data,
                        net='darknet19',
                        detmodel='yolo',
                        max_iters=10000,
                        expid=expid,
                        snapshot=500,
                        restore_snapshot_iter=-1,
                        remove_bb_label=label,
                        remove_bb_label_prob=prob,
                        monitor_train_only=monitor_train_only,
                        add_image_rule=add_image_rule,
                        gpus='4,5,6,7',
                        ovthresh=[0,0.1,0.2,0.3,0.4,0.5])
                if is_report:
                    c = CaffeWrapper(**kwargs)
                    perf = None
                    try:
                        perf = c.best_model_perf()
                    except:
                        continue
                    if perf:
                        if label not in report:
                            report[label] = {}
                        for k in perf['overall']:
                            if prob not in report[label]:
                                report[label][prob] = {}
                            report[label][prob][k] = perf['overall'][k]['class_ap'][label]
                else:
                    all_task.append(kwargs)
    if is_report:
        caption = 'remove bb with coco' if add_coco_image else 'remove bb'
        if add_coco_image:
            probs = [0.9, 1]
        else:
            probs = [0, 0.9, 1]
        buf = print_m_table(report, [all_labels[:10], probs], [map(str,
            [0,0.1,0.2,0.3,0.4,0.5])], caption)
        fname = \
        '/raid/jianfw/work/voc20/darknet19/remove_bb{}'.format('_add_coco_image'.format(add_coco_image)
                if add_coco_image else '')
        write_to_file(buf, '{}_1.tex'.format(fname))
        buf = print_m_table(report, [all_labels[10:], probs], [map(str,
            [0,0.1,0.2,0.3,0.4,0.5])], caption)
        write_to_file(buf, '{}_2.tex'.format(fname))

    if not is_report:
        #task_processor('-1', all_task[0])
        b = BatchProcess(all_resource, all_task, task_processor)
        b.run()

def test_vggstyle():
    v = VGGStyle()
    n = caffe.NetSpec()
    n.data = caffe.layers.Layer()
    v.add_body(n)

def classification_task():
    '''
    deprecated. merging to yolo_master
    '''
    #all_net_stage_sizes.append([7, 6, 6])
    #all_net_stage_sizes.append([15, 14, 14])
    ps = []
    all_task = []
    net_bn_last = False
    net_triangle_feature_degree = 1
    bnloss_type = None
    #bnloss_type = 'whitten'
    #bnloss_type = '1_mag'
    bnloss_r = 2
    if bnloss_type is not None:
        bnloss = {'param_str': json.dumps({'loss_type': bnloss_type,
            'r': bnloss_r})}
    else:
        bnloss = {}
    weight_decay_before_n = True
    net_localnormalization = False
    #data = 'cifar10'
    #data = 'cifar10_first5'
    #data = 'mnist'
    data = 'imagenet2012'
    sample_label = 1
    sample_image = 1
    net_shuffle_unit = False
    inception_crop = False
    weight_decay=0.0001
    kernel_active = 0
    shuffle_net_add_shift_before_first_1x1 = True
    shrink_group_if_group_e_out = True
    net_version = 1
    stem_1x1_only = False
    all_kernel_sizes = [[3, 3, 3]]
    net_stem_channels = 27
    num_conv_per_block = 2
    if data == 'imagenet2012':
        net = 'vggstyle'
        net = 'resnet18'
        net_version = 2
        #net = 'templatenet'
        #template_net = './aux_data/shufflenet_1x_g3_deploy.prototxt'
        #net = 'resnet10'
        sample_label = 0.9
        sample_label = 1
        sample_image = 1
        all_net_base_channel = [72, 54, 36]
        all_net_base_channel = [54]
        #all_net_base_channel = [144]
        num_conv_per_block = 2
        all_net_stage_sizes = []
        all_net_stage_sizes += [[2, 4, 2]]
        #all_net_stage_sizes += [[1, 2, 1]]
        #all_net_stage_sizes += [[4, 8, 4]]
        net_stem_channels = 27
        all_kernel_sizes = [[3, 3, 3]]
        #all_kernel_sizes = [[5, 3, 3]]
        #all_kernel_sizes = [[1,1,1]]
        #all_kernel_sizes = [[3,3,3]]
        ave_pool_reduce = False
        #stem_1x1_only = True
        stem_1x1_only = False
        net_shuffle_unit = True
        #net_shuffle_unit = False
        shuffle_net_fully3x3 = False
        if net_shuffle_unit:
            all_net_stage_sizes = [[4, 8, 4]]
            all_net_base_channel = [144]
            net_stem_channels = 24
            #kernel_active = 0
            weight_decay=0.00005
            net = 'vggstyle'

        #inception_crop = False
    elif data.startswith('cifar10') or data.startswith('mnist'):
        net = 'vggstyle'
        all_net_base_channel = [32]
        all_net_stage_sizes = []
        all_net_stage_sizes.append([3, 2, 2])

    #channels_factor = 2
    #kernel_active_skip = 1
    channels_factor = 1
    channels_factor_skip = 0
    #monitor_train_only = True
    monitor_train_only = False
    use_shift = True
    #use_shift = False
    net_stage_sizes = all_net_stage_sizes[0]

    dataset_ops = [{'op':'select_top', 'num_top': 5}]
    bkg_idx = 5
    dataset_ops = [
            {'op': 'mask_background', 'old_label_idx': [5,6,7,8,9],
        'new_label_idx': bkg_idx}
        #{'op': 'select_top', 'num_top': 60000}
        ]
    init_from = {'data': 'cifar10_first5', 
            'net': 'vggstyle', 
            'expid': 'B_bn_32_3_2_2_weightdecay0.0001_add_res',
            'type': 'min_l2', 
            'new_data': 'cifar10_second5'}
    #init_from = {}
    dataset_ops = []
    max_iters = 10000
    max_iters = '90e'
    #max_iters = 5000
    lr_policy = 'step'
    #lr_policy = 'multifixed'
    stagelr=[0.01, 0.01],
    stageiter=[10000, 100000],
    if data.startswith('mnist'):
        lr_policy = 'multifixed'
        stagelr = (np.asarray([0.1, 0.01, 0.01]) * 1).tolist()
        max_iters = '120e'
        stageiter = ['30e', '60e', '90e']

    burn_in='',
    burn_in_power=1,

    kwargs_template = dict(data=data,
            net=net,
            monitor_train_only=monitor_train_only,
            #solver_debug_info=True,
            #force_train=True,
            detmodel='classification',
            max_iters=max_iters,
            lr_policy=lr_policy,
            base_lr = 0.1,
            stepsize = 2000,
            dataset_ops=dataset_ops,
            use_pretrained=False,
            snapshot=500,
            stagelr=stagelr,
            stageiter=stageiter,
            crop_type=caffe.params.TsvData.InceptionStyle,
            inception_crop_kl='./data/imagenet2012/kl.txt',
            #restore_snapshot_iter=-1,
            #last_ip_lr_mult=0.1,
            #last_ip_decay = 1,
            display=100)

    first_layer_as_1x1 = False
    if net == 'resnet10' or net == 'resnet18' or net == 'vggstyle' and net_version == 2:
        kwargs_template['cls_add_global_pooling'] = True

    if data == 'imagenet2012':
        kwargs_template['effective_batch_size'] = 256
        if sample_label >= 1 and sample_image >= 1 or \
                sample_label == 0.9 and sample_image == 1:
            kwargs_template['max_iters'] = 450000
            kwargs_template['lr_policy'] = 'step'
            kwargs_template['snapshot'] = 5000
            kwargs_template['base_lr'] = 0.1
            kwargs_template['stepsize'] = 100000
        if sample_label == 0.1 and sample_image == 1:
            kwargs_template['max_iters'] = 45000
            kwargs_template['lr_policy'] = 'step'
            kwargs_template['snapshot'] = 5000
            kwargs_template['base_lr'] = 0.1
            kwargs_template['stepsize'] = 10000
    lr_base = 1
    if data.startswith('cifar10'):
        lr_base = 1
        kwargs_template['lr_policy'] = 'multifixed'
        kwargs_template['stagelr'] = [0.01 * lr_base,
                0.001 * lr_base, 0.0001 * lr_base]

    padding = 0
    net_l2, net_bn = False, True
    if data == 'mnist':
        padding = 4
        net_bn = True
    net_conv_sym = None
    kernel_active_skip = 1
    if stem_1x1_only and kernel_active == 1:
        kernel_active_skip = 0
    #kernel_active_type = 'SEQ_1x1'
    kernel_active_type = 'SEQ'
    #kernel_active_type = 'UNIFORM_1x1'
    net_add_res = True
    if data == 'mnist':
        ensure_mnist_tsv()
    bkg_plus_softmax = True
    bkg_plus_softmax = False
    bkg_plus_num_classes = 10
    for net_stage_sizes in all_net_stage_sizes:
        for net_base_channel in all_net_base_channel:
            for kernel_sizes in all_kernel_sizes:
                #for sigmoid_loss in [True, False]:
                for sigmoid_loss in [False]:
                    ave_pool_reduce = False
                    #for monitor_train_only in [True]:
                    expid = 'B'
                    kwargs = copy.deepcopy(kwargs_template)
                    if sample_label < 1 or sample_image < 1:
                        expid = '{}_sl{}si{}'.format(expid,
                                sample_label, sample_image)
                        kwargs['dataset_ops'] = [{
                            'op': 'sample',
                            'sample_label': sample_label,
                            'sample_image': sample_image}]
                    if net == 'vggstyle' and net_version != 1:
                        expid = '{}_netV{}'.format(expid, net_version)
                        kwargs['net_version'] = net_version
                    if net == 'vggstyle' and net_version != 1 and net_stem_channels != 24:
                        expid = '{}_stemChannel{}'.format(expid,
                                net_stem_channels)
                        kwargs['net_stem_channels'] = net_stem_channels
                    if net_shuffle_unit and \
                            net == 'vggstyle' and \
                            net_version == 2:
                        expid = '{}_shuffleNet'.format(expid)
                        kwargs['net_shuffle_unit'] = net_shuffle_unit
                        if shuffle_net_fully3x3:
                            expid = '{}_Fully3x3'.format(expid)
                            kwargs['shuffle_net_fully3x3'] = shuffle_net_fully3x3
                        if shuffle_net_add_shift_before_first_1x1:
                            expid = '{}_shiftFirst1x1'.format(expid)
                            kwargs['shuffle_net_add_shift_before_first_1x1']=True
                    if net == 'vggstyle' and net_version != 1 and not net_shuffle_unit and num_conv_per_block != 1:
                        expid = '{}_numConvPerBlock{}'.format(expid,
                                num_conv_per_block)
                        kwargs['net_num_conv_per_block'] = num_conv_per_block
                    if net == 'templatenet':
                        expid = '{}_{}'.format(expid,
                                op.basename(template_net))
                        kwargs['template_net'] = template_net
                    if not inception_crop:
                        kwargs.pop('crop_type', None)
                        kwargs.pop('inception_crop_kl', None)
                        if not data.startswith('cifar10'):
                            expid = '{}_noInceptCrop'.format(expid)
                    if net == 'vggstyle':
                        if data.startswith('cifar10'):
                            kwargs['stageiter'] = [32000, 48000,
                                    1000000]
                        if net_conv_sym:
                            expid = '{}_{}'.format(expid,
                                    net_conv_sym)
                            kwargs['net_conv_sym'] = net_conv_sym
                        if net_l2:
                            expid = '{}_{}'.format(expid,
                                    net_l2)
                            kwargs['net_l2'] = net_l2
                        if net_bn:
                            expid = '{}_bn'.format(expid)
                            kwargs['net_bn'] = net_bn
                        kwargs['net_base_channel'] = net_base_channel
                        expid = '{}_{}'.format(expid,
                                net_base_channel)
                        kwargs['net_stage_sizes'] = net_stage_sizes
                        expid = '{}_{}'.format(expid,
                                '_'.join(map(str,
                                    net_stage_sizes)))
                        if padding != 0:
                            expid = '{}_padding{}'.format(expid,
                                    padding)
                            kwargs['transform_param_padding']=\
                                padding
                        if weight_decay != 0.0005:
                            expid = '{}_weightdecay{}'.format(
                                    expid,
                                    weight_decay)
                            kwargs['weight_decay'] = weight_decay
                        if lr_base != 1:
                            expid = '{}_lrbase{}'.format(expid,
                                    lr_base)
                        if not weight_decay_before_n:
                            expid = '{}_freeconv'.format(expid)
                            kwargs['weight_decay_before_n'] = weight_decay_before_n
                        if not net_shuffle_unit:
                            if net_add_res and net_version == 1:
                                expid = '{}_add_res'.format(expid)
                                kwargs['net_add_res'] = net_add_res
                            if not net_add_res and net_version == 2:
                                expid = '{}_noRes'.format(expid)
                                kwargs['net_add_res'] = net_add_res
                        if data.startswith('cifar10'):
                            kwargs['new_width'] = 32
                            kwargs['new_height'] = 32
                            kwargs['crop_size'] = 32
                            kwargs['mean_value'] = [127.5, 127.5,
                                    127.5]
                            kwargs['effective_batch_size'] = 128
                            kwargs['max_iters'] = 64000
                            kwargs['snapshot'] = 1000
                        elif data == 'mnist':
                            kwargs['new_width'] = 28
                            kwargs['new_height'] = 28
                            kwargs['crop_size'] = 28
                            kwargs['mean_value'] = [127.5, 127.5,
                                    127.5]
                            kwargs['effective_batch_size'] = 128
                    elif net == 'resnet10' or net == 'resnet18':
                        if weight_decay != 0.0005:
                            expid = '{}_weightdecay{}'.format(
                                    expid,
                                    weight_decay)
                            kwargs['weight_decay'] = weight_decay
                    if net_triangle_feature_degree != 1:
                        expid = '{}_ntriangle{}'.format(expid,
                                net_triangle_feature_degree)
                        kwargs['net_triangle_feature_degree']=net_triangle_feature_degree
                    if len(bnloss) > 0:
                        if bnloss_type == 'whitten':
                            expid = '{}_bnloss{}'.format(expid, bnloss_type)
                        else:
                            expid = '{}_bnloss{}_{}'.format(expid,
                                    bnloss_type, bnloss_r)
                        kwargs['bnloss'] = bnloss
                    if net_localnormalization:
                        expid = '{}_localnorm'.format(expid)
                        kwargs['net_localnormalization'] = net_localnormalization
                    if kernel_active > 0:
                        expid = '{}_kernelActive{}'.format(expid,
                                kernel_active)
                        kwargs['kernel_active'] = kernel_active 
                        if kernel_active_skip != 0:
                            expid = '{}_activeSkip{}'.format(expid,
                                    kernel_active_skip)
                            kwargs['kernel_active_skip'] = kernel_active_skip
                        if use_shift:
                            expid = '{}_useShiftN'.format(expid)
                            kwargs['use_shift'] = use_shift
                        if kernel_active_type != 'SEQ':
                            expid = '{}_{}'.format(expid, kernel_active_type)
                            kwargs['kernel_active_type'] = kernel_active_type
                        if shrink_group_if_group_e_out:
                            expid = '{}_shrinkGroup'.format(expid)
                            kwargs['shrink_group_if_group_e_out'] = shrink_group_if_group_e_out
                    if channels_factor != 1:
                        expid = '{}_channelsFactor{}'.format(expid,
                                channels_factor)
                        kwargs['channels_factor'] = channels_factor
                        if channels_factor_skip != 0:
                            expid = '{}_skip{}'.format(expid,
                                    channels_factor_skip)
                            kwargs['channels_factor_skip'] = \
                                    channels_factor_skip
                    if first_layer_as_1x1:
                        expid = '{}_first1x1'.format(expid)
                        kwargs['first_layer_as_1x1'] = first_layer_as_1x1
                    if ave_pool_reduce:
                        expid = '{}_avePoolReduce'.format(expid)
                        kwargs['ave_pool_reduce'] = ave_pool_reduce
                    if not all(k == 3 for k in kernel_sizes):
                        expid = '{}_ks{}'.format(expid, '.'.join(map(str,
                            kernel_sizes)))
                        kwargs['kernel_sizes'] = kernel_sizes
                    if stem_1x1_only:
                        expid = '{}_stem1x1'.format(expid)
                        kwargs['stem_1x1_only'] = stem_1x1_only
                    if len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'mask_background':
                        expid = '{}_maskBackground{}_{}'.format(expid,
                                '.'.join(map(str,
                                    dataset_ops[0]['old_label_idx'])),
                                dataset_ops[0]['new_label_idx'])
                        kwargs['dataset_ops'] = dataset_ops
                    if bkg_plus_softmax:
                        expid = '{}_bkgPlusSoftmax{}'.format(expid,
                                bkg_plus_num_classes)
                        kwargs['bkg_plus_softmax'] = bkg_plus_softmax
                        kwargs['bkg_plus_num_classes'] = bkg_plus_num_classes
                        kwargs['bkg_idx'] = bkg_idx
                    if sigmoid_loss:
                        expid = '{}_Sigmoid'.format(expid)
                        kwargs['sigmoid_loss'] = sigmoid_loss
                    if len(init_from) != 0:
                        kwargs['init_from'] = copy.deepcopy(init_from)
                    kwargs['expid'] = expid
                    if data == 'imagenet2012':
                        kwargs['predict_style'] = 'tsvdatalayer'
                    all_task.append(kwargs)
           
    machines = get_machine()
    cluster4 = machines['clusters4']
    cluster8 = machines['clusters8']
    all_gpus = [[0,1,2,3], [4,5,6,7]]
    all_gpus = [[4, 5, 6, 7]]
    all_gpus = [[0, 1]]
    #all_gpus = [[0, 1, 2, 3]]
    #all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[-1] * 16]
    all_resource = []
    if data.startswith('cifar10') or data == 'mnist':
        all_gpus = [[0], [1], [2], [3], [4], [5], [6], [7]]
        all_resource += [(vigs[1], g) for g in all_gpus]
        #all_resource += [(vigs[0], g) for g in all_gpus]
        #all_resource += [(clusters8[0], g) for g in all_gpus]
        #all_resource += [(clusters8[1], g) for g in all_gpus]
    else:
        #all_resource += [(vig1, g) for g in all_gpus]
        #all_resource += [(vig2, [0, 1, 2, 3])]
        #all_resource += [(vig2, [0, 1])]
        #all_resource += [(vig2, [2, 3])]
        #all_resource += [(vig2, [4, 5])]
        #all_resource += [(vig2, [4, 5, 6, 7])]
        #all_resource += [(cluster8, [0, 1])]
        #all_resource += [(cluster8, [2, 3])]
        #all_resource += [(cluster8, [4, 5])]
        #all_resource += [(cluster8, [6, 7])]
        all_resource += [(cluster4, [0, 1, 2, 3])]
        all_resource += [(cluster8, [0, 1, 2, 3])]
        all_resource += [(cluster8, [4, 5, 6, 7])]
        #all_resource += [(vig2, [6, 7])]
        #all_resource += [(vig1, [0, 1, 2, 3])]
        #all_resource += [(vig1, [0, 1])]
        #all_resource += [(vig1, [2, 3])]
        #all_resource += [(vig1, [4, 5])]
        #all_resource += [(vig1, [6, 7])]
        #all_resource += [(cluster4, [-1] * 8)]
        #all_resource += [(cluster2_1, [0, 1])]
        #all_resource += [(cluster2_2, [0, 1])]
        pass
    #all_resource += [(vig1, g) for g in all_gpus]

    logging.info(pformat(all_resource))
    logging.info(pformat(all_task))
    logging.info('#task: {}'.format(len(all_task)))
    logging.info('#resource: {}'.format(len(all_resource)))
    
    debug = False
    return
    #debug = True
    #all_task[0]['expid'] = 'lei'
    #all_task[0]['skip_genprototxt'] = True
    #all_task[0]['force_predict'] = True
    #all_task[0]['force_evaluate'] = True
    def run():
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
        return
        if not monitor_train_only:
            for t in all_task:
                t['monitor_train_only'] = True
            #for i, r in enumerate(all_resource):
                #all_resource[i] = (r[0], [-1] * 4)
            b = BatchProcess(all_resource, all_task, task_processor)
            #b._availability_check = False
            b.run()

    if debug:
        #all_task[0]['max_iters'] = 10000
        #all_task[0]['max_iters'] = 10
        #all_task[0]['display'] = 1
        #all_task[0]['solver_debug_info'] = True
        #all_task[0]['force_train'] = True
        all_task[0]['debug_train'] = True
        #all_task[0]['effective_batch_size'] = 16
        #all_task[0]['expid'] = 'lei'
        #all_task[0]['skip_genprototxt'] = True
        all_task[0]['debug_detect'] = True
        all_task[0]['force_predict'] = True
        #all_task[0]['expid'] = 'debug'
        task_processor(({}, [4]), all_task[0])
    return
    run()

def cifar():
    '''
    deprecated. use classification_task()
    '''
    #plot_remove_bb_result()
    #test_vggstyle()
    #all_net_base_channel = [16, 32, 64]
    #all_net_base_channel = [16, 32, 64, 128]
    all_net_base_channel = [16, 32]
    #all_net_base_channel = [128]
    all_net_stage_sizes = []
    #all_net_stage_sizes.append([1, 1, 1, 1])
    #all_net_stage_sizes.append([4, 3, 2, 1])
    #all_net_stage_sizes.append([1, 2, 3, 4])
    #all_net_stage_sizes.append([2, 2, 2, 2])
    all_net_stage_sizes.append([3, 2, 2])
    all_net_stage_sizes.append([7, 6, 6])
    all_net_stage_sizes.append([15, 14, 14])
    #all_net_stage_sizes.append([11, 10, 10])
    #all_net_stage_sizes.append([19, 18, 18])
    #all_net_stage_sizes.append([4, 4, 4])
    #all_net_stage_sizes.append([8, 8, 8])
    #all_net_stage_sizes.append([16, 16, 16])
    all_padding = [4]
    #all_net_conv_sym = [None, 'LeftRight']
    all_net_conv_sym = [None]
    #all_net_conv_sym = ['LeftRight']
    #all_net_conv_sym = [None]
    ps = []
    all_task = []
    net_bn_last = False
    net_triangle_feature_degree = 1
    bnloss_type = None
    #bnloss_type = '1_mag'
    bnloss_r = 1
    if bnloss_type is not None:
        bnloss = {'param_str': json.dumps({'loss_type': bnloss_type,
            'r': bnloss_r})}
    else:
        bnloss = {}
    weight_decay_before_n = True
    net_localnormalization = False
    data = 'cifar10'
    #data = 'imagenet2012'
    sample_label = 1
    sample_image = 1

    if data == 'imagenet2012':
        net = 'resnet10'
        all_net_stage_sizes = [7, 6, 6]
        all_net_base_channel = [16]
    else:
        net = 'vggstyle'

    #channels_factor = 2
    #kernel_active_skip = 1
    channels_factor = 1
    channels_factor_skip = 0
    #monitor_train_only = True
    monitor_train_only = False
    use_shift = True
    #use_shift = False
    net_stage_sizes = all_net_stage_sizes[0]
    kwargs_template = dict(data=data,
            net=net,
            monitor_train_only=monitor_train_only,
            layer_wise_reduce=False,
            #solver_debug_info=True,
            #force_train=True,
            detmodel='classification',
            use_pretrained=False,
            #restore_snapshot_iter=-1,
            #solver_debug_info=True,
            display=100,
            #source_shuffle='tmp_shuffle.txt',
            ovthresh=[0,0.1,0.2,0.3,0.4,0.5])
    kernel_active = 0
    for net_stage_sizes in all_net_stage_sizes:
        #for kernel_active in [0, 1, 2, 4, 6]:
        for residual_loss in [True, False]:
            #for kernel_active_skip in [1, 2, 3]:
            for kernel_active_skip in [1]:
                for net_base_channel in all_net_base_channel:
                    for net_conv_sym in all_net_conv_sym:
                        for padding in all_padding:
                            #for net_l2, net_bn in [[True, False], [False, True]]:
                            for net_l2, net_bn in [[False, True]]:
                            #for net_l2, net_bn in [[True, True]]:
                            #for net_l2, net_bn in [[False, True]]:
                                #for net_add_res in [False, True]:
                                for net_add_res in [False]:
                                    #for monitor_train_only in [True]:
                                    weight_decay=0.0001
                                    lr_base = 10
                                    expid = 'B'
                                    kwargs = copy.deepcopy(kwargs_template)
                                    if sample_label < 1 or sample_image < 1:
                                        expid = '{}_sl{}si{}'.format(expid,
                                                sample_label, sample_image)
                                        kwargs['dataset_ops'] = [{
                                            'op': 'sample',
                                            'sample_label': sample_label,
                                            'sample_image': sample_image}]
                                    if net == 'vggstyle':
                                        if data == 'cifar10':
                                            kwargs['stageiter'] = [32000, 48000,
                                                    1000000]
                                        if net_conv_sym:
                                            expid = '{}_{}'.format(expid,
                                                    net_conv_sym)
                                            kwargs['net_conv_sym'] = net_conv_sym
                                        if net_l2:
                                            expid = '{}_{}'.format(expid,
                                                    net_l2)
                                            kwargs['net_l2'] = net_l2
                                        if net_bn:
                                            expid = '{}_bn'.format(expid)
                                            kwargs['net_bn'] = net_bn
                                        kwargs['net_base_channel'] = net_base_channel
                                        expid = '{}_{}'.format(expid,
                                                net_base_channel)
                                        kwargs['net_stage_sizes'] = net_stage_sizes
                                        expid = '{}_{}'.format(expid,
                                                '_'.join(map(str,
                                                    net_stage_sizes)))
                                        if padding != 0:
                                            expid = '{}_padding{}'.format(expid,
                                                    padding)
                                            kwargs['transform_param_padding']=\
                                                padding
                                        if weight_decay != 0.0005:
                                            expid = '{}_weightdecay{}'.format(
                                                    expid,
                                                    weight_decay)
                                            kwargs['weight_decay'] = weight_decay
                                        if lr_base != 1:
                                            expid = '{}_lrbase{}'.format(expid,
                                                    lr_base)
                                        kwargs['lr_policy'] = 'multifixed'
                                        kwargs['stagelr'] = [0.01 * lr_base,
                                                0.001 * lr_base, 0.0001 *
                                                lr_base]
                                        if not weight_decay_before_n:
                                            expid = '{}_freeconv'.format(expid)
                                            kwargs['weight_decay_before_n'] = weight_decay_before_n
                                        if net_add_res:
                                            expid = '{}_add_res'.format(expid)
                                            kwargs['net_add_res'] = net_add_res
                                        kwargs['new_width'] = 32
                                        kwargs['new_height'] = 32
                                        kwargs['crop_size'] = 32
                                        kwargs['mean_value'] = [127.5, 127.5,
                                                127.5]
                                        kwargs['effective_batch_size'] = 128
                                        kwargs['max_iters'] = 64000
                                        kwargs['snapshot'] = 1000
                                    elif net == 'resnet10':
                                        kwargs['cls_add_global_pooling'] = True
                                        kwargs['crop_type'] = \
                                            caffe.params.TsvData.InceptionStyle
                                        kwargs['inception_crop_kl'] = \
                                                './data/imagenet2012/kl.txt'
                                        kwargs['effective_batch_size'] = 256
                                        kwargs['max_iters'] = 450000
                                        kwargs['lr_policy'] = 'step'
                                        kwargs['snapshot'] = 5000
                                        kwargs['base_lr'] = 0.1
                                        kwargs['stepsize'] = 100000
                                        if weight_decay != 0.0005:
                                            expid = '{}_weightdecay{}'.format(
                                                    expid,
                                                    weight_decay)
                                            kwargs['weight_decay'] = weight_decay
                                    
                                    if net_triangle_feature_degree != 1:
                                        expid = '{}_ntriangle{}'.format(expid,
                                                net_triangle_feature_degree)
                                        kwargs['net_triangle_feature_degree']=net_triangle_feature_degree
                                    if len(bnloss) > 0:
                                        expid = '{}_bnloss{}_{}'.format(expid,
                                                bnloss_type, bnloss_r)
                                        kwargs['bnloss'] = bnloss
                                    if net_localnormalization:
                                        expid = '{}_localnorm'.format(expid)
                                        kwargs['net_localnormalization'] = net_localnormalization
                                    if kernel_active > 0:
                                        expid = '{}_kernelActive{}'.format(expid,
                                                kernel_active)
                                        kwargs['kernel_active'] = kernel_active 
                                        if kernel_active_skip != 0:
                                            expid = '{}_activeSkip{}'.format(expid,
                                                    kernel_active_skip)
                                            kwargs['kernel_active_skip'] = kernel_active_skip
                                        if use_shift:
                                            expid = '{}_useShiftN'.format(expid)
                                            kwargs['use_shift'] = use_shift

                                    if channels_factor != 1:
                                        expid = '{}_channelsFactor{}'.format(expid,
                                                channels_factor)
                                        kwargs['channels_factor'] = channels_factor
                                        if channels_factor_skip != 0:
                                            expid = '{}_skip{}'.format(expid,
                                                    channels_factor_skip)
                                            kwargs['channels_factor_skip'] = \
                                                    channels_factor_skip
                                    if residual_loss:
                                        expid = '{}_resLoss'.format(expid)
                                        kwargs['residual_loss'] = residual_loss
                                    kwargs['expid'] = expid
                                    if data == 'imagenet2012':
                                        kwargs['predict_style'] = 'tsvdatalayer'
                                    all_task.append(kwargs)
    
    vigs, clusters8 = get_machine()
    all_gpus = [[0,1,2,3], [4,5,6,7]]
    all_gpus = [[4, 5, 6, 7]]
    all_gpus = [[0, 1, 2, 3]]
    all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[-1] * 16]
    all_resource = []
    if data == 'cifar10':
        all_gpus = [[0], [1], [2], [3], [4], [5], [6], [7]]
        #all_gpus = [[0], [1]]
        #all_resource += [(vig2, g) for g in all_gpus]
        all_resource += [(vig2, g) for g in all_gpus]
        #all_resource += [(vig2, g) for g in all_gpus]
        #all_resource += [(cluster4, g) for g in all_gpus]
    else:
        #all_resource += [(vig1, g) for g in all_gpus]
        all_resource += [(vig2, g) for g in all_gpus]
        #all_resource += [(cluster4, g) for g in all_gpus]
        pass
    #all_resource += [(vig1, g) for g in all_gpus]

    logging.info(pformat(all_resource))
    logging.info(pformat(all_task))
    logging.info('#task: {}'.format(len(all_task)))
    logging.info('#resource: {}'.format(len(all_resource)))
    
    debug = False
    #debug = True
    #all_task[0]['expid'] = 'lei'
    #all_task[0]['skip_genprototxt'] = True
    #all_task[0]['force_predict'] = True
    #all_task[0]['force_evaluate'] = True

    if debug:
        all_task[0]['max_iters'] = 1
        all_task[0]['force_train'] = True
        all_task[0]['debug_train'] = True
        all_task[0]['effective_batch_size'] = 16
        #all_task[0]['expid'] = 'lei'
        #all_task[0]['skip_genprototxt'] = True
        #all_task[0]['debug_detect'] = True
        #all_task[0]['force_predict'] = True
        #all_task[0]['expid'] = 'debug'
        task_processor(({}, [4]), all_task[0])
    else:
        #b = BatchProcess(all_resource, all_task, task_processor)
        ##b._availability_check = False
        #b.run()
        pass

def print_label_order():
    source_tsv = '/home/jianfw/code/quickdetection/data/cifar100/train.tsv'
    from process_tsv import TSVTransformer
    t = TSVTransformer()
    from pprint import pprint
    t.ReadProcess(source_tsv, lambda row: pprint(row[1]))

def test_yolo9000_on_imagenet():
    # input to test imagenet
    #test_source = './data/imagenet/test.tsv'
    #test_proto_file = './output/imagenet/reproduce/yolo9000_map.prototxt'
    #model_param = './output/imagenet/reproduce/yolo9000.caffemodel'
    #outtsv_file = './output/imagenet/yolo9000/predict_result.tsv'

    # input to test voc20
    test_source = './data/voc20/test.tsv'
    test_proto_file = './output/imagenet/reproduce/yolo9000_map_voc20.prototxt'
    model_param = './output/imagenet/reproduce/yolo9000.caffemodel'
    outtsv_file = './output/voc20/yolo9000/predict_result.tsv'

    # output
    kwargs = {}
    kwargs['gpus'] = [0,1,2,3,4,5,6,7]
    #kwargs['gpus'] = [-1]

    dl8 = {'username': 'jianfw',
            'ip':'10.196.44.185',
            '-p': 30824,
            'data': '/work/data/qd_data_cluster',
            'output': '/work/work/qd_output'}
    sync_qd(dl8)
    kwargs['caffenet'] = test_proto_file
    kwargs['caffemodel'] = model_param
    kwargs['intsv_file'] = test_source
    kwargs['key_idx'] = 0
    kwargs['img_idx'] = 2
    kwargs['pixel_mean'] = [0]
    kwargs['scale'] = 1
    kwargs['outtsv_file'] = outtsv_file
    kwargs['force_evaluate'] = True

    remote_python_run(tsvdet, kwargs, dl8)
    #tsvdet(**kwargs)

    deteval(truth=test_source, dets=outtsv_file, **kwargs)

def compare_log_for_multibin():
    base_log_file = \
        './output/voc20_darknet19_A/log_rank_0_20170830-004642.28044-main'
    #m_log_file = \
    #'./output/voc20_darknet19_A_multibin_wh_0_13_16/log_rank_0_20170830-014945.4961-main-log'
    #m_log_file = \
        #'./output/voc20_darknet19_A_no_rescore/log_rank_0_20170901-050025.29878'
    #m_log_file = \
        #'./output/voc20_darknet19_A_ignore/log_rank_0_20170902-162331.13588'
    #base_log_file = \
        #'./output/voc20_darknet19_A_objkl/log_rank_0_20170901-235236.43803-main'
    #m_log_file = \
        #'./output/voc20_darknet19_A_objkl/log_rank_0_20170901-235236.43803-main'
    #m_log_file = \
        #'./output/voc20_darknet19_A_norescore_objkl/log_rank_0_20170902-212125.25687'
    #m_log_file = \
        #'./output/voc20_darknet19_A_objkl_1/log_rank_0_20170902-211623.24394'
    #m_log_file = \
        #'./output/voc20_darknet19_A_norescore_objkl_objscale1/log_rank_0_20170902-225400.31140'
    #m_log_file = \
        #'./output/voc20_darknet19_A_norescore_objkl_objscale1_xykl/log_rank_0_20170903-002305.43846'
    #base_log_file = \
        #'./output/voc20_darknet19_A_norescore_objonly/log_rank_0_20170903-121454.75045'
    #m_log_file = \
        #'./output/voc20_darknet19_A_norescore_objkl_objscale1_objonly/log_rank_0_20170903-095328.69673'
    #m_log_file = \
            #'./output/voc20_darknet19_A_explinearwh/log_rank_0_20170905-105638.60564'
    #m_log_file = \
            #'./output/voc20_darknet19_A_nonobjtoiou/log_rank_0_20170905-134144.3943'
    #m_log_file = \
            #'./output/voc20_darknet19_A_norescore_objkl_objonly/log_rank_0_20170905-142422.7187'
    #m_log_file = \
            #'./output/voc20_darknet19_A_norescore_ignore_objonly/log_rank_0_20170905-153622.21653'
    m_log_file = \
            './output/voc20_darknet19_A_norescore_objkl_objonly_around1/log_rank_0_20170905-233131.12041'
    m_log_file = \
            './output/voc20_darknet19_A_norescore_objonly_around1/log_rank_0_20170905-163121.33666'
    m_log_file = \
            './output/voc20_darknet19_A_nolosssmallthaniou/log_rank_0_20170907-134223.76052'
    m_log_file = \
            './output/voc20_darknet19_A_objcaparound/log_rank_0_20170907-171233.42879'
    m_log_file = \
            './output/voc20_darknet19_A_fixedtarget/log_rank_0_20170910-131427.40616'


    #xys, whs, objs, cls = parse_yolo_log(base_log_file)
    #m_xys, m_whs, m_objs, m_cls = parse_yolo_log(m_log_file)
    all_ious, all_probs, all_obj, all_anyobj, all_recall, all_count = \
        parse_yolo_log_acc(base_log_file)
    all_obj = np.asarray(all_obj)
    all_count = np.asarray(all_count)
    all_anyobj = np.asarray(all_anyobj)

    m_ious, m_probs, m_obj, m_noobj, m_recall, m_count = \
        parse_yolo_log_acc(m_log_file)

    m_obj = np.asarray(m_obj)
    m_count = np.asarray(m_count)
    m_noobj = np.asarray(m_noobj)

    #plt.plot(all_recall)
    #plt.plot(m_recall)

    #fig = plt.figure()
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].plot(all_ious[2:])
    ax[0, 0].plot(m_ious[2:])
    ax[0, 0].legend(('base', 'm'), loc='best')
    ax[0, 0].grid()
    ax[0, 0].set_title('average iou on the target')

    ax[0, 1].plot(all_probs[2:])
    ax[0, 1].plot(m_probs[2:])
    ax[0, 1].grid()
    ax[0, 1].legend(('base', 'm'), loc='best')
    ax[0, 1].set_title('average classification prob on the target')

    ax[0, 2].plot(all_recall[2:])
    ax[0, 2].plot(m_recall[2:])
    ax[0, 2].grid()
    ax[0, 2].legend(('base', 'm'), loc='best')
    ax[0, 2].set_title('recall')

    ax[1, 0].plot(all_obj[2:])
    ax[1, 0].plot(m_obj[2:])
    ax[1, 0].grid()
    ax[1, 0].legend(('base', 'm'), loc='best')
    ax[1, 0].set_title('average objectiveness on the target')

    ax[1, 1].plot((all_anyobj[5:] * 16 * 5 * 13 * 13 - all_obj[5:] * all_count[5:]) /
            (16 * 5 * 13 * 13 - all_count[5:]))
    ax[1, 1].plot((m_noobj[5:] * 16 * 5 * 13 * 13 - m_obj[5:] * m_count[5:]) /
            (16 * 5 * 13 * 13 - m_count[5:]))
    ax[1, 1].grid()
    ax[1, 1].legend(('base', 'm'), loc='best')
    ax[1, 1].set_title('average objectiveness on non-targets')
    #plt.plot(np.asarray(xys) / 5)
    #plt.plot(m_xys)

    #plt.plot(whs)
    #plt.plot(m_whs)

    #plt.plot(np.asarray(objs[10: ]) / 5)
    #plt.plot(m_objs[10: ])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #fig.savefig('/home/jianfw/a.png')

    plt.show()

def visualize_multibin():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    data = 'voc20'
    test_set = TSVDataset('voc20')
    label_names = test_set.load_labelmap()

    expid = 'A_multibinXY0_0.96875_16'

    c = CaffeWrapper(data='voc20', net='darknet19', detmodel='yolo',
            expid=expid, yolo_extract_target_prediction=True)

    m = construct_model(c._path_env['solver'],
            c._path_env['train_proto_file'])

    tsv_file = c._predict_file(m)

    train_net = load_net(c._path_env['train_proto_file'])

    for l in train_net.layer:
        if l.type == 'TsvBoxData':
            l.include.pop()
            l.include.add(phase=caffe.proto.caffe_pb2.TEST)
            l.transform_param.mirror = False
            #l.data_param.prefetch = 1
            l.box_data_param.jitter = 0
            l.box_data_param.hue = 0
            l.box_data_param.exposure = 1
            l.box_data_param.saturation = 1
            l.box_data_param.random_scale_min = 1
            l.box_data_param.random_scale_max = 1
            l.tsv_data_param.batch_size = 1
            #l.tsv_data_param.source = test_set.get_test_tsv_file()
            tsv_file = l.tsv_data_param.source
        if l.type == 'RegionLoss':
            l.top.append('conf_debug')
            l.loss_weight.append(1)
            l.loss_weight.append(0)
            l.region_loss_param.debug_info = 1
            multibin_wh = l.region_loss_param.multibin_wh
            multibin_xy = l.region_loss_param.multibin_xy
            if multibin_wh:
                multibin_wh_count = l.region_loss_param.multibin_wh_count
                multibin_wh_low = l.region_loss_param.multibin_wh_low
                multibin_wh_high = l.region_loss_param.multibin_wh_high
            if multibin_xy:
                multibin_xy_low = l.region_loss_param.multibin_xy_low
                multibin_xy_high = l.region_loss_param.multibin_xy_low
                multibin_xy_count = l.region_loss_param.multibin_xy_count

    if multibin_wh:
        multibin_wh_step =(multibin_wh_high-multibin_wh_low)/(multibin_wh_count-1)
    if multibin_xy:
        multibin_xy_step = (multibin_xy_high-multibin_xy_low)/(multibin_xy_count-1)

    debug_proto = c._path_env['train_proto_file'] + '.debug'

    write_to_file(str(train_net), debug_proto)

    net = caffe.Net(debug_proto, m.model_param, caffe.TEST)

    rows = tsv_reader(tsv_file)
    #while True:
    for j, row in enumerate(rows):
        logging.info(j)
        net.forward()

        i = 0
        im = (net.blobs['data'].data[i].transpose((1, 2, 0)) + np.asarray(m.mean_value).reshape(1, 1, 3)).astype(np.uint8)
        #im = im.transpose((2, 0, 1))
        label = net.blobs['label'].data[i]
        label = label.reshape((30, 5))
        label_valid = label[:, 0] != 0
        label = label[label_valid, :]
        gt_bbox = label[:, :-1] * im.shape[1]
        x0 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2
        x1 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
        y0 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2
        y1 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
        gt_bbox = np.hstack((x0[:, np.newaxis], y0[:, np.newaxis], x1[:, np.newaxis], y1[:, np.newaxis]))
        gt_label = [label_names[int(l)] for l in label[:, -1]]
        gt_im = im.copy()
        conf = net.blobs['conf_debug'].data[i]
        draw_bb(gt_im, gt_bbox, gt_label)

        f, ax = plt.subplots(4, len(gt_label) + 1)
        #f, ax = plt.subplots(2, 1)
        ax[0, 0].imshow(cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB),
                interpolation='none')
        ax[0, 0].set_title('network input')

        ax[1, 0].imshow(cv2.cvtColor(img_from_base64(row[2]),
            cv2.COLOR_BGR2RGB), interpolation='none')
        ax[1, 0].set_title('original image')
        plt.show()

        w_s = 2
        w_e = int(w_s + multibin_wh_count)
        h_s = int(w_e)
        h_e = int(h_s + multibin_wh_count)
        c_obj = int(2 + 2 * multibin_wh_count)
        cls_s = int(c_obj + 1)
        cls_e = int(cls_s + 20)
        im_obj = (conf[c_obj, :, :] * 255).astype(np.uint8).copy()
        ax[1, 0].imshow(im_obj, cmap='gray')

        for g in range(len(gt_label)):
            b = gt_bbox[g]
            target_i = (b[0] + b[2]) / 2 / im.shape[1] * im_obj.shape[1]
            target_i_res = target_i - (int)(target_i)
            target_i = int(target_i)
            target_j = (b[1] + b[3]) / 2 / im.shape[1] * im_obj.shape[1]
            target_j_res = target_j - (int)(target_j)
            target_j = int(target_j)

            if multibin_wh:
                peak_w = ((b[2] - b[0]) / im.shape[1] * im_obj.shape[1] - multibin_wh_low) / multibin_wh_step
                peak_h = ((b[3] - b[1])/im.shape[1]*im_obj.shape[1] - multibin_wh_low) / multibin_wh_step

            curr_o = conf[c_obj, target_j, target_i]
            curr_w = conf[w_s:w_e, target_j, target_i]
            curr_h = conf[h_s:h_e, target_j, target_i]
            curr_cls = conf[cls_s:cls_e, target_j, target_i]

            assert abs(sum(curr_w) - 1) < 0.01, sum(curr_w)
            assert abs(sum(curr_h) - 1) < 0.01, sum(curr_h)

            ax[0, g + 1].plot(curr_w)
            ax[0, g + 1].set_xlabel('{}:{:.2f}'.format(gt_label[g], peak_w))
            ax[0, g + 1].grid()
            ax[1, g + 1].plot(curr_h)
            ax[1, g + 1].grid()
            ax[1, g + 1].set_xlabel('{}:{:.2f}: {:.2f}'.format(gt_label[g], peak_h,
                curr_o))

        plt.show()

def forward_net(net, data_blob, img_info, blob_label, m):
    net.blobs['data'].reshape(1, *data_blob.shape)
    net.blobs['data'].data[...] = data_blob.reshape(1, *data_blob.shape)
    net.blobs['im_info'].reshape(1,2)
    net.blobs['im_info'].data[...] = img_info
    net.blobs['label'].data[...] = blob_label
    net.forward()

    im = (net.blobs['data'].data[i].transpose((1, 2, 0)) + np.asarray(m.mean_value).reshape(1, 1, 3)).astype(np.uint8)
    conf = net.blobs['conf_debug'].data[0]
    tp = net.blobs['target_prediction'].data[0]
    label = net.blobs['label'].data[i]

    return im, label, conf, tp

def move_image_label(data_blob, blob_label):
    right_move = 16
    height = data_blob.shape[-2]
    width = data_blob.shape[-1]
    for j in range(height):
        for i in range(width - 1, right_move - 1, -1):
            for c in range(3):
                data_blob[c, j, i] = data_blob[c, j, i - right_move]
        for i in range(0, right_move):
            for c in range(3):
                data_blob[c, j, i] = 0

    for i in range(blob_label.size / 5):
        x, y, w, h = blob_label[0, i * 5 : i * 5 + 4]
        if x == 0:
            break
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        x0 = x0 + 16. / width
        x1 = x1 + 16. / width
        x0, y0, x1, y1 = min(x0, 1), min(y0, 1), min(x1, 1), min(y1, 1)
        blob_label[0, i * 5 : i * 5 + 4] = (x0 + x1) / 2., (y0 + y1) / 2., \
                x1 - x0, y1 - y0


def visualize_multibin2():
    data = 'voc20'
    test_set = TSVDataset('voc20')
    label_names = test_set.load_labelmap()

    #expid = 'A_multibinXY16'
    expid = 'A'

    c = CaffeWrapper(data='voc20', net='darknet19', detmodel='yolo',
            expid=expid, yolo_extract_target_prediction=True)

    m = construct_model(c._path_env['solver'],
            c._path_env['test_proto_file'])

    test_net = load_net(c._path_env['test_proto_file'])
    for l in test_net.layer:
        if l.type == 'RegionOutput':
            l.bottom
            l.loss_weight.append(1)
            l.loss_weight.append(0)
            l.region_loss_param.debug_info = 1
            multibin_wh = l.region_output_param.multibin_wh
            multibin_xy = l.region_output_param.multibin_xy
            coords = 0
            if multibin_wh:
                multibin_wh_count = l.region_output_param.multibin_wh_count
                multibin_wh_low = l.region_output_param.multibin_wh_low
                multibin_wh_high = l.region_output_param.multibin_wh_high
                coords = coords + 2 * multibin_wh_count
            else:
                coords = coords + 2
            if multibin_xy:
                multibin_xy_low = l.region_output_param.multibin_xy_low
                multibin_xy_high = l.region_output_param.multibin_xy_high
                multibin_xy_count = l.region_output_param.multibin_xy_count
                coords = coords + 2 * multibin_xy_count
            else:
                coords = coords + 2

    if multibin_wh:
        multibin_wh_step =(multibin_wh_high-multibin_wh_low)/(multibin_wh_count-1)
    if multibin_xy:
        multibin_xy_step = (multibin_xy_high-multibin_xy_low)/(multibin_xy_count-1)

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(c._get_test_proto_file(m), m.model_param, caffe.TEST)

    rows = tsv_reader(test_set.get_train_tsv())
    move = True
    #while True:
    for j, row in enumerate(rows):
        i = 0

        data_blob, img_info, blob_label = prepare_net_input(img_from_base64(row[2]),
                m.mean_value, 416,
                gt_labels=json.loads(row[1]), label_map=label_names,
                yolo_max_truth=300)
        
        if move:
            move_image_label(data_blob, blob_label)

        net.blobs['data'].reshape(1, *data_blob.shape)
        net.blobs['data'].data[...]=data_blob.reshape(1, *data_blob.shape)
        net.blobs['im_info'].reshape(1,2)
        net.blobs['im_info'].data[...] = img_info
        net.blobs['label'].data[...] = blob_label
        net.forward()
        im = (net.blobs['data'].data[i].transpose((1, 2, 0)) + np.asarray(m.mean_value).reshape(1, 1, 3)).astype(np.uint8)

        #im = im.transpose((2, 0, 1))
        label = net.blobs['label'].data[i]
        label = label.reshape((-1, 5))
        label_valid = label[:, 0] != 0
        label = label[label_valid, :]
        gt_bbox = label[:, :-1] * im.shape[1]
        x0 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2
        x1 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
        y0 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2
        y1 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
        gt_bbox = np.hstack((x0[:, np.newaxis], y0[:, np.newaxis], x1[:, np.newaxis], y1[:, np.newaxis]))
        gt_label = [label_names[int(l)] for l in label[:, -1]]

        conf = net.blobs['conf_debug'].data[0]
        tp = net.blobs['target_prediction'].data[0]
        for g in range(len(gt_label)):
            f, ax = plt.subplots(2, 2)

            b = gt_bbox[g]
            target_i = (b[0] + b[2]) / 2 / im.shape[1] * 13
            target_i_res = target_i - (int)(target_i)
            target_i = int(target_i)
            target_j = (b[1] + b[3]) / 2 / im.shape[1] * 13
            target_j_res = target_j - (int)(target_j)
            target_j = int(target_j)

            # grid image
            assert tp[g, 4, 0] == target_i
            assert tp[g, 5, 0] == target_j
            target_n = tp[g, 6, 0]
            offset = target_n * (coords + 21)
            offset = int(offset)
            

            grid_im = im.copy()
            for _j in range(13):
                cv2.line(grid_im, (0, _j * 32), (415, _j * 32), (0, 0, 255))
            for _i in range(13):
                cv2.line(grid_im, (_i * 32, 0), (_i * 32, 415), (0, 0, 255))
            for extra_j in range(-1, 2):
                for extra_i in range(-1, 2):
                    if multibin_xy:
                        idx_x = np.argmax(conf[offset:(multibin_xy_count+offset),
                            target_j + extra_j, target_i + extra_i])
                        pred_x_in_im = idx_x * multibin_xy_step + target_i + \
                                extra_i
                        idx_y = np.argmax(conf[(offset+multibin_xy_count): (offset + 2 *
                            multibin_xy_count), target_j + extra_j, target_i +
                            extra_i])
                        pred_y_in_im = idx_y * multibin_xy_step + target_j + \
                            extra_j
                    else:
                        pred_x_in_im = (conf[int(target_n * (coords + 21)),
                                             target_j + extra_j, target_i +
                                             extra_i] + target_i + extra_i)
                        pred_y_in_im = conf[int(target_n * (coords + 21)) + 1,
                                target_j + extra_j,                             target_i + extra_i] + target_j + extra_j

                    pred_x_in_im = pred_x_in_im * 32
                    pred_x_in_im = int(pred_x_in_im)
                    pred_y_in_im = pred_y_in_im * 32
                    pred_y_in_im = int(pred_y_in_im)
                    if multibin_xy:
                        pred_x_conf = conf[offset:(offset+multibin_xy_count),
                                target_j + extra_j, target_i + extra_i]
                        pred_y_conf = conf[(offset+multibin_xy_count): (offset + 2*multibin_xy_count),
                                target_j + extra_j, target_i + extra_i]
                        ideal_x_conf = np.zeros(multibin_xy_count)
                        ideal_x_pos = (target_i_res - multibin_xy_low) / multibin_xy_step
                        ideal_x_pos = int(ideal_x_pos + 0.5)
                        ideal_x_conf[ideal_x_pos] = 1
                        ideal_y_conf = np.zeros(multibin_xy_count)
                        ideal_y_pos = (target_j_res - multibin_xy_low) / multibin_xy_step
                        ideal_y_pos = int(ideal_y_pos + 0.5)
                        ideal_y_conf[ideal_y_pos] = 1
                        
                        if extra_i == 0 and extra_j == 0:
                            ax[0, 1].plot(pred_x_conf, '-o')
                            ax[0, 1].plot((ideal_x_pos, ideal_x_pos), (0,
                                pred_x_conf[ideal_x_pos]), '-o')
                            ax[0, 1].set_xlim(0, multibin_xy_count)
                            ax[0, 1].grid()
                            ax[0, 1].set_title('x')

                            ax[1, 1].plot(pred_y_conf, '-o')
                            ax[1, 1].plot((ideal_y_pos, ideal_y_pos), (0,
                                pred_y_conf[ideal_y_pos]), '-o')
                            ax[1, 1].set_xlim(0, multibin_xy_count)
                            ax[1, 1].grid()
                            ax[1, 1].set_title('y')
                    else:
                        pred_x_res = conf[offset + 0, target_j + extra_j,
                                target_i + extra_i]
                        pred_y_res = conf[offset + 1, target_j + extra_j,
                                target_i + extra_i]
                        if extra_i == 0 and extra_j == 0:
                            ax[0, 1].plot((pred_x_res, pred_x_res), (0, 1), '-o')
                            ax[0, 1].plot((target_i_res, target_i_res), (0, 0.5), '-*')

                            ax[1, 1].plot((pred_y_res, pred_y_res), (0, 1), '-o')
                            ax[1, 1].plot((target_j_res, target_j_res), (0, 0.5), '-*')
                    origin_from = ((target_i + extra_i) * 32, (target_j +
                        extra_j) * 32)
                    gt_to = (int((b[0]+b[2])/2), int((b[1]+b[3])/2))
                    if extra_i == 0 and extra_j == 0:
                        cv2.line(grid_im, origin_from, gt_to, (0, 0, 255))
                        cv2.rectangle(grid_im, (int(b[0]), int(b[1])),
                                (int(b[2]), int(b[3])), (0, 0, 255), 2)
                    cv2.line(grid_im, origin_from, (pred_x_in_im, pred_y_in_im), (255,
                        0, 0), 2)

            ax[0, 0].imshow(cv2.cvtColor(grid_im, cv2.COLOR_BGR2RGB))
            ax[0, 0].set_title('grid input')

            ax[1, 0].imshow(cv2.cvtColor(img_from_base64(row[2]),
                cv2.COLOR_BGR2RGB), interpolation='none')
            ax[1, 0].set_title('original image')
            figure_fname = '/home/jianfw/work/tmp/{}{}_{}_{}.png'.format(expid,
                    j, g, move)
            plt.savefig(figure_fname)
            plt.close(f)
        if j > 10:
            break

def print_8(ss):
    overall_s = None
    for s in ss:
        if overall_s is None:
            overall_s = s / sum(np.asarray(s))
        else:
            overall_s += s / sum(np.asarray(s))
    keys = ['000', '001', '010', '011', '100', '101', '110', '111']
    overall_s /= len(ss)
    logging.info(pformat(zip(keys, map(lambda x: '{:.2f}'.format(x * 100),
        overall_s))))

def test_():
    populate_dataset_details('icdar_e2e_2015_focused')
    #folder = '/home/jianfw/code/pr/quickdetection/output/tax700_debug_darknet19_testTaxonomy_bb_only'
    #proto = op.join(folder, 'test.prototxt')
    #model = op.join(folder, 'snapshot', 'model_iter_2.caffemodel')
    #net = load_binary_net(model)
    #import ipdb;ipdb.set_trace(context=15)


def study_target():
    fname = \
            './output/voc20_darknet19_A/snapshot/model_iter_10000.caffemodel.voc20.extract_target.predict'
    fname = \
            './output/voc20_darknet19_A_decay0/snapshot/model_iter_10000.caffemodel.voc20.extract_target.predict'
    fname = \
            './output/voc20_darknet19_A_align0/snapshot/model_iter_10000.caffemodel.voc20.extract_target.predict'
    #expid = 'A'
    #expid = 'A_multibinXY0_32'
    expid = 'A_multibinXY32'
    #expid = 'A_multibinXY32_objscale0_noobjScale0_clsScale0'
    c = CaffeWrapper(data='voc20', net='resnet34', detmodel='yolo', 
            expid=expid, yolo_extract_target_prediction=True)
    model = construct_model(c._path_env['solver'], c._path_env['test_proto_file'])
    fname = c._predict_file(model)
    region_output_param = load_net(c._path_env['test_proto_file']).layer[-1].region_output_param
    multibin_xy = region_output_param.multibin_xy
    multibin_xy_low = region_output_param.multibin_xy_low
    multibin_xy_high = region_output_param.multibin_xy_high
    multibin_xy_count = region_output_param.multibin_xy_count
    coords_xy = 2
    if multibin_xy:
        multibin_xy_step = (multibin_xy_high - multibin_xy_low) / (multibin_xy_count - 1)
        coords_xy = 2 * multibin_xy_count
    rows = tsv_reader(fname)
    gt_xs = []
    gt_ys = []
    gt_ws = []
    gt_hs = []
    pred_xs = []
    t_target_xs = {'s': [], 'm': [], 'l': []}
    t_target_ys = {'s': [], 'm': [], 'l': []}
    t_pred_xs = {'s': [], 'm': [], 'l': []}
    pred_ys = []
    t_pred_ys = {'s': [], 'm': [], 'l': []}
    pred_ws = []
    pred_hs = []
    target_xs = []
    before_pred_xs, before_pred_ys = [], []
    target_ys = []
    target_ws, target_hs = [], []
    pred_exp_ws = []
    pred_exp_hs = []
    biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
    for row in rows:
        target = pkl.loads(base64.b64decode(row[1]))
        for r in range(target.shape[0]):
            if target[r, 1, 0] == 0 and target[r, 0, 0] == 0:
                continue
            gt_xs.append(target[r, 0, 0])
            gt_ys.append(target[r, 1, 0])
            gt_w = target[r, 2, 0]
            gt_ws.append(gt_w)
            gt_h = target[r, 3, 0]
            gt_hs.append(gt_h)
            curr_target_x = target[r, 0, 0] * 13 - target[r, 4, 0]
            curr_target_y = target[r, 1, 0] * 13 - target[r, 5, 0]
            target_xs.append(curr_target_x)
            target_ys.append(curr_target_y)
            if not multibin_xy:
                curr_pred_x = target[r, 7, 0]
                before_pred_xs.append(-np.log(1.0 / target[r, 7, 0] - 1))
                curr_pred_y = target[r, 8, 0]
                before_pred_ys.append(-np.log(1.0 / target[r, 8, 0] - 1))
            else:
                pred_x_hist = target[r, 7 : 7 + multibin_xy_count, 0]
                assert np.abs(np.sum(pred_x_hist) - 1) < 0.001
                idx_x = np.argmax(pred_x_hist)
                curr_pred_x = idx_x * multibin_xy_step + multibin_xy_low
                pred_y_hist = target[r, 7 + multibin_xy_count : 7 + 2 *
                    multibin_xy_count, 0]
                assert np.abs(np.sum(pred_y_hist) - 1) < 0.001
                idx_y = np.argmax(pred_y_hist)
                curr_pred_y = idx_y * multibin_xy_step + multibin_xy_low
            pred_xs.append(curr_pred_x)
            pred_ys.append(curr_pred_y)
            if gt_w * gt_h < 1. / 13 / 13.:
                t_target_xs['s'].append(curr_target_x)
                t_target_ys['s'].append(curr_target_y)
                t_pred_xs['s'].append(curr_pred_x)
                t_pred_ys['s'].append(curr_pred_y)
            elif gt_w * gt_h < 9. / 13 / 13:
                t_target_xs['m'].append(curr_target_x)
                t_target_ys['m'].append(curr_target_y)
                t_pred_xs['m'].append(curr_pred_x)
                t_pred_ys['m'].append(curr_pred_y)
            else:
                t_target_xs['l'].append(curr_target_x)
                t_target_ys['l'].append(curr_target_y)
                t_pred_xs['l'].append(curr_pred_x)
                t_pred_ys['l'].append(curr_pred_y)
            
            n = int(target[r, 6, 0])
            pred_ws.append(target[r, 7 + coords_xy, 0])
            pred_exp_ws.append(np.exp(target[r, 7 + coords_xy, 0]) * biases[2 * n])
            pred_hs.append(target[r, 8 + coords_xy, 0])
            pred_exp_hs.append(np.exp(target[r, 8 + coords_xy, 0]) * biases[2 * n + 1])
            check_best_iou(biases, target[r, 2, 0] * 13, target[r, 3, 0] * 13, n)
            target_ws.append(target[r, 2, 0] * 13 / biases[2 * n])
            target_hs.append(target[r, 3, 0] * 13 / biases[2 * n + 1])

    _, ax = plt.subplots(6, 2)
    bins = 32 if multibin_xy_count == 0 else multibin_xy_count
    r = 0
    ax[r, 0].hist(target_xs, bins=bins)
    ax[r, 0].set_title('gt-x * 13 - target_x')
    ax[r, 1].hist(target_ys, bins=bins)
    ax[r, 1].set_title('gt-y * 13 - target_y')
    r = r + 1

    ax[r, 0].hist(pred_xs, bins=bins)
    ax[r, 0].set_title('pred x')
    ax[r, 1].hist(pred_ys, bins=bins)
    ax[r, 1].set_title('pred y')
    r = r + 1

    #ax[r, 0].hist(t_target_xs['s'], bins=bins)
    #ax[r, 0].set_title('s target x')
    #ax[r, 1].hist(t_target_ys['s'], bins=bins)
    #ax[r, 1].set_title('s target y')
    #r = r + 1

    ax[r, 0].hist(t_pred_xs['s'], bins=bins)
    ax[r, 0].set_title('s pred x')
    ax[r, 1].hist(t_pred_ys['s'], bins=bins)
    ax[r, 0].set_title('s pred y')
    r = r + 1

    ax[r, 0].hist(t_pred_xs['m'], bins=bins)
    ax[r, 0].set_title('m target x')
    ax[r, 1].hist(t_pred_ys['m'], bins=bins)
    ax[r, 0].set_title('m target y')
    r = r + 1
    
    ax[r, 0].hist(t_pred_xs['l'], bins=bins)
    ax[r, 0].set_title('l pred x')
    ax[r, 1].hist(t_pred_ys['l'], bins=bins)
    ax[r, 0].set_title('l pred y')
    r = r + 1

    ax[r, 0].hist(before_pred_xs)
    ax[r, 1].hist(before_pred_ys)

    logging.info(np.mean(np.abs(np.asarray(target_xs) - np.asarray(pred_xs))))
    logging.info(np.mean(np.abs(np.asarray(target_ys) - np.asarray(pred_ys))))
    logging.info(np.mean(np.abs(np.asarray(t_target_xs['s']) -
        np.asarray(t_pred_xs['s']))))
    logging.info(np.mean(np.abs(np.asarray(t_target_ys['s']) -
        np.asarray(t_pred_ys['s']))))
    logging.info(np.mean(np.abs(np.asarray(t_target_xs['m']) -
        np.asarray(t_pred_xs['m']))))
    logging.info(np.mean(np.abs(np.asarray(t_target_ys['m']) -
        np.asarray(t_pred_ys['m']))))
    logging.info(np.mean(np.abs(np.asarray(t_target_xs['l']) -
        np.asarray(t_pred_xs['l']))))
    logging.info(np.mean(np.abs(np.asarray(t_target_ys['l']) -
        np.asarray(t_pred_ys['l']))))

    #r = r + 1
    #ax[r, 0].hist(gt_ws)
    #ax[r, 0].set_title('gt w')
    #ax[r, 1].hist(gt_hs)
    #ax[r, 1].set_title('gt h')
    #r = r + 1
    #ax[r, 0].hist(gt_ws)
    #ax[r, 0].set_title('exp(pred) * b w')
    #ax[r, 1].hist(gt_hs)
    #ax[r, 1].set_title('exp(pred) * b h')
    #r = r + 1
    #ax[r, 0].hist(map(lambda x: np.log(x), target_ws))
    #ax[r, 0].set_title('log(gt/b) w')
    #ax[r, 1].hist(map(lambda x: np.log(x), target_hs))
    #ax[r, 1].set_title('log(gt/b) h')
    #r = r + 1
    #ax[r, 0].hist(pred_ws)
    #ax[r, 0].set_title('pred w')
    #ax[r, 1].hist(pred_hs)
    #ax[r, 1].set_title('pred h')
    plt.show()

    #ax[0, 0].hist(ws)
    #ax[0, 1].hist(hs)
    #ax[1, 0].hist(pws)
    #ax[1, 1].hist(phs)
    #plt.show()


def flops(out_dim, stride, repeat, in_channel, out_channel, g):
    bottle_channel = out_channel if stride == 1 else out_channel - in_channel
    x = out_channel/4 * out_dim * stride * out_dim * stride * in_channel / g
    x = x + out_channel / 4 * out_dim  * out_dim * 9
    x = x+bottle_channel*out_dim*out_dim*out_channel/4/g
    #if stride != 1:
        #x = x + in_channel * out_dim * out_dim * 9
    #else:
        #x = x + out_dim * out_dim * out_channel
    return x * repeat

def mobile_net():
    x = 112 * 112 * 32 * 3 * 3 * 3
    x = x + 32 * 112 * 112 * 3 * 3
    x = x + 64 * 112 * 112 * 32
    x = x + 64 * 56 * 56 * 3 * 3 
    x = x + 128 * 56 * 56 * 64
    x = x + 128 * 56 * 56 * 3 * 3
    x = x + 128 * 56 * 56 * 128
    x = x + 128 * 28 * 28 * 3 * 3
    x = x + 256 * 28 * 28 * 128
    x = x + 256 * 28 * 28 * 3 * 3 
    x = x + 256 * 28 * 28 * 256
    x = x + 256 * 14 * 14 * 3 * 3
    x = x + 512 * 14 * 14 * 256
    x = x + 5 * (512 * 14 * 14 * 9 + 512 * 14 * 14 * 512)
    x = x + 512 * 7 * 7 * 3 * 3
    x = x + 1024 * 7 * 7 * 512
    x = x + 1024 * 7 * 7 * 3 * 3
    x = x + 1024 * 7 * 7 * 1024
    x = x + 1024 * 1000
    #x = x + 1024 * 7 * 7
    logging.info(x / 1000000.)

def all_flops():
    x = 0
    x = 112 * 112 * 24 * 9 * 3
    x = x + flops(28, 2, 1, 24, 144, 1)
    x = x + flops(28, 1, 3, 144, 144, 1)
    x = x + flops(14, 2, 1, 144, 288, 1)
    x = x + flops(14, 1, 7, 288, 288, 1)
    x = x + flops(7, 2, 1, 288, 576, 1)
    x = x + flops(7, 1, 3, 576, 576, 1)
    x = x + 576 * 49
    x = x + 1000 * 576
    logging.info(x / 1000000.)

def low_shot_checking():
    low_shot_label = 'aeroplane'
    #low_shot_label = 'bicycle'
    num_train = 1
    expid = 'A_lowShot.{}.{}{}'.format(low_shot_label, 
            num_train, 
            '_noBiasLastConv')
    expid = 'A_extraconv3_3_3_3_lowShot.{}.{}_fullGpu_AngularRegulizer_noBiasconf'.format(low_shot_label, 
            num_train)

    #expid = 'A_decay0'
    #expid = 'A'

    data = 'voc20'
    dataset = TSVDataset(data)
    full_labels = dataset.load_labelmap()
    num_classes = len(full_labels)
    low_shot_novel_idx = [full_labels.index(low_shot_label)]
    low_shot_base_idx = [i for i in range(num_classes) if i not in
        low_shot_novel_idx]
    c = CaffeWrapper(data=data, 
            #net='resnet34', 
            net='darknet19_448', 
            detmodel='yolo',
            expid=expid)

    perf = c.best_model_perf()
    class_ap = perf['overall']['0.5']['class_ap']

    # construct by net
    model = construct_model(c._path_env['solver'], 
            c._path_env['test_proto_file'], is_last=True)

    net = caffe.Net(c._path_env['test_proto_file'], model.model_param,
            caffe.TEST)
    
    net_blob_weight = np.squeeze(net.params['conf'][0].data)
    assert len(net.params['conf']) == 1
    import ipdb;ipdb.set_trace()

    logging.info('{}-{}'.format(expid, low_shot_label))
    logging.info('novel ap: {}'.format(np.mean(class_ap[low_shot_label])))
    logging.info('base ap: {}'.format(np.mean([class_ap[k] for k in class_ap.keys() if k !=
        low_shot_label])))

    #model = load_binary_net(model.model_param)
    #weight_blob = model.layer[-2].blobs[0]

    #weight = np.asarray(weight_blob.data).reshape(weight_blob.shape.dim)
    #weight = np.squeeze(weight)

    weight = net_blob_weight
    #logging.info(np.linalg.norm(weight[:] - net_blob_weight[:]))

    weight_norm = np.linalg.norm(weight, axis=1)
    #weight_norm = weight 
    num_anchor = 5
    num_classes = c._get_num_classes()
    assert len(weight_norm) == 5 * num_classes
    for i in range(num_anchor):
        idx_from = i * num_classes
        idx_to = idx_from + num_classes
        wn = weight_norm[idx_from : idx_to]

        base_wn = wn[low_shot_base_idx]
        logging.info('base: {}-{}; {}-{}'.format(np.mean(base_wn), 
            np.sqrt(np.var(base_wn)), np.min(base_wn), np.max(base_wn)))

        novel_wn = wn[low_shot_novel_idx]
        logging.info('novel: {}-{}; {}-{}'.format(np.mean(novel_wn), 
            np.sqrt(np.var(novel_wn)), np.min(novel_wn), np.max(novel_wn)))

def convert_layout(o_last_conv):
    n_last_conv = np.zeros(o_last_conv.shape)
    num_anchor = o_last_conv.shape[1] / 25
    for b in range(o_last_conv.shape[0]):
        for a in range(num_anchor):
            # x
            n_last_conv[b, a, :, :] = o_last_conv[b, a * 25 + 0, :, :]
            # y
            n_last_conv[b, num_anchor + a, :, :] = o_last_conv[b, a * 25 + 1, :, :]
            # w
            n_last_conv[b, 2 * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 2, :, :]
            # h
            n_last_conv[b, 3 * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 3, :, :]
            # o
            n_last_conv[b, 4 * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 4, :, :]
            # cls
            n_last_conv[b, 5 * num_anchor + a * 20: 5 * num_anchor + a * 20 +
                    20, :, :] = o_last_conv[b, a * 25 + 5 : a * 25 + 25, :, :]
    return n_last_conv

def check_yolo_test_full_gpu():
    new_layer_proto = './aux_data/new_yolo_test.prototxt'
    old_layer_proto = './aux_data/old_yolo_test.prototxt'
    caffe.set_mode_gpu()
    o = caffe.Net(old_layer_proto, caffe.TEST)
    n = caffe.Net(new_layer_proto, caffe.TEST)
    #np.random.seed(779)
    batch_size = 1
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    im_info = [200, 200]
    
    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['im_info'].data[...] = im_info
    o.forward()
    o_prob = o.blobs['prob'].data[0]
    o_bbox = o.blobs['bbox'].data[0]

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['im_info'].data[...] = im_info
    n.forward()
    n_bbox = n.blobs['bbox'].data[0]
    n_bbox = n_bbox.reshape(-1, n_bbox.shape[-1])
    n_prob = n.blobs['prob'].data[0]
    n_prob = n_prob.reshape(-1, n_prob.shape[-1])

    x = np.abs(n_bbox[:] - o_bbox[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_bbox.shape))
    logging.info(n_bbox.reshape(-1)[idx])
    logging.info(o_bbox.reshape(-1)[idx])

    y = np.abs(n_prob[:] - o_prob[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_prob.shape))
    logging.info(n_prob.reshape(-1)[idx])
    logging.info(o_prob.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()

def check_yolo_full_gpu():
    new_layer_proto = './aux_data/new_layers.prototxt'
    old_layer_proto = './aux_data/old_layer.prototxt'
    caffe.set_mode_gpu()
    n = caffe.Net(new_layer_proto, caffe.TRAIN)
    o = caffe.Net(old_layer_proto, caffe.TRAIN)
    #np.random.seed(779)
    batch_size = 6
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    label = np.zeros((6, 150))
    for i in range(batch_size):
        for j in range(15):
            label[i, 5 * j + 0] = np.random.rand()
            label[i, 5 * j + 1] = np.random.rand()
            label[i, 5 * j + 2] = np.random.rand()
            label[i, 5 * j + 3] = np.random.rand()
            label[i, 5 * j + 4] = 5
    
    iter_number = 158000
    #iter_number = 0

    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['label'].data[...] = label
    o.params['region_loss'][0].data[0] = iter_number
    o.params['region_loss']
    o.forward()
    o.backward()
    o_diff= o.blobs['last_conv'].diff

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['label'].data[...] = label
    n.params['region_target'][0].data[0] = iter_number
    n.forward()
    n.backward()
    n_diff = n.blobs['last_conv'].diff

    on_diff = convert_layout(o_diff)
    x = np.abs(n_diff[:] - on_diff[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_diff.shape))
    logging.info(n_diff.reshape(-1)[idx])
    logging.info(on_diff.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()

def study_loss_per_cat():
    fname = './output/imagenet_darknet19_A_noreorg_noextraconv/snapshot/model_iter_500200.caffemodel.evaluate_loss_per_cls'
    tree_file = './aux_data/yolo/9k.tree'
    noffset_idx, noffset_parentidx, noffsets = load_label_parent(tree_file)
    loss = pkl.loads(read_to_buffer(fname))
    ave_prob = loss[0, 1, :, 0] / (loss[0, 0, :, 0] + 0.001)
    ave_loss = loss[0, 2, :, 0] / (loss[0, 0, :, 0] + 0.001)
    nick_names = [get_nick_name(noffset_to_synset(no)) for no in noffsets]
    write_to_file('\n'.join(['\t'.join(map(str, x)) for x in zip(noffsets, nick_names, ave_prob, ave_loss,
        loss[0, 0, :, 0])]), '/home/jianfw/work/abc.txt')
    #return
    idx = sorted(range(loss.shape[2]), key=lambda x: ave_prob[x])
    _, ax = plt.subplots(2, 2)
    ax[0, 0].plot(ave_prob[idx])
    ax[0, 0].grid()
    ax[0, 1].plot(ave_loss[idx])
    ax[0, 1].grid()
    ax[1, 0].plot(loss[0, 0, idx, 0])
    ax[1, 0].grid()
    #plt.plot(loss[0, 2, :, 0] / loss[0, 0, :, 0])
    plt.show()

def compare():
    expids = ['A_fullGpu_burnIn5e.1', 'A_fullGpu']
    for expid in expids:
        c = CaffeWrapper(data='voc20', net='resnet34', detmodel='yolo', expid=expid)
        curr = c.best_model_perf()
        curr['overall']['0.5']['map']

def design_massive():
    input_size = np.asarray(640)
    strides = np.power(2, range(1, 9))
    feature_map_size = input_size / strides
    logging.info(strides)
    logging.info(feature_map_size)
    feature_map_area = feature_map_size ** 2
    logging.info(feature_map_area)
    logging.info(range(2, 10))
    logging.info(np.sum(feature_map_area) * 15)

def extract_labels(data='coco2017', labels=['cell phone', 'person'],
        out_data='coco_phone_person'):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)
    splits = ['train', 'test']

    label_mapper = {l: l for l in labels}

    for split in splits:
        if op.isfile(out_dataset.get_data(split)):
            continue
        inverted = dataset.load_inverted_label(split)
        idxes = [inverted[l] for l in labels]
        out_idx = list(set([i for idx in idxes for i in idx]))
        tsv = TSVFile(dataset.get_data(split))
        def gen_rows():
            for i in out_idx:
                row = tsv.seek(i)
                rects = json.loads(row[1])
                convert_one_label(rects, label_mapper)
                assert len(rects) > 0
                row[1] = json.dumps(rects)
                yield row
        tsv_writer(gen_rows(), out_dataset.get_data(split))
    
    if not op.isfile(out_dataset.get_labelmap_file()):
        write_to_file('\n'.join(labels), out_dataset.get_labelmap_file())

    populate_dataset_details(out_data)

def calculate_ap(predicts, gts, cat):
    predict_rect = [(key, rect) for key in predicts
            for rect in predicts[key] if rect['class'] == cat]
    predict_rect.sort(key=lambda x: -x[1]['conf'])
    corrects = np.zeros(len(predict_rect))
    dets = set()
    for i, (key, rect) in enumerate(predict_rect):
        curr_gt = gts[key]
        ious = [calculate_iou(rect['rect'], g['rect']) for g in curr_gt if
                g['class'] == cat]
        if len(ious) == 0:
            continue
        idx = np.argmax(ious)
        if ious[idx] > 0.3 and (key, idx) not in dets:
            corrects[i] = 1
            dets.add((key, idx))

    total = len([() for key in gts for rect in
        gts[key] if rect['class'] == cat])
    return calculate_ap_by_true_list(corrects, total)

def calculate_ap_by_true_list(corrects, total):
    import qd_common
    return qd_common.calculate_ap_by_true_list(corrects, total)

def calculate_image_ap(predicts, gts):
    '''
    a list of rects
    '''
    import qd_common
    return qd_common.calculate_image_ap(predicts, gts)
    #matched = [False] * len(gts)
    #corrects = np.zeros(len(predicts))
    #for j, p in enumerate(predicts):
        #for i, g in enumerate(gts):
            #if not matched[i]:
                #iou = calculate_iou(p, g)
                #if iou > 0.3:
                    #matched[i] = True
                    #corrects[j] = 1
    #return calculate_ap_by_true_list(corrects, len(gts))

def get_confusion_matrix_by_predict_file(full_expid, 
        predict_file, threshold, test_data_split='test'):
    import tsv_io
    return tsv_io.get_confusion_matrix_by_predict_file(full_expid, 
        predict_file, threshold, test_data_split)
    #test_data = parse_test_data(predict_file)
    #predicts, _ = load_labels(op.join('output', full_expid, 'snapshot', predict_file))

    ## load the gt
    #test_dataset = TSVDataset(test_data)
    #test_label_file = test_dataset.get_data(test_data_split, 'label')
    #gts, label_to_idx = load_labels(test_label_file)

    ## calculate the confusion matrix
    #confusion_pred_gt = {}
    #confusion_gt_pred = {}
    #update_confusion_matrix(predicts, gts, threshold, 
            #confusion_pred_gt, 
            #confusion_gt_pred)

    #return {'predicts': predicts, 
            #'gts': gts, 
            #'confusion_pred_gt': confusion_pred_gt, 
            #'confusion_gt_pred': confusion_gt_pred,
            #'label_to_idx': label_to_idx}

def get_confusion_matrix(data, net, test_data, expid, threshold=0.2, **kwargs):
    import yolotrain
    return yolotrain.get_confusion_matrix(data, net, test_data, expid,
            threshold, **kwargs)
    #logging.info('deprecated: use get_confusion_matrix_by_predict_file')
    #c = CaffeWrapper(data=data, net=net, 
            #test_data=test_data,
            #yolo_test_maintain_ratio = True,
            #expid=expid,
            #**kwargs)
    
    ## load predicted results
    #model = c.best_model()
    #predict_file = c._predict_file(model)

    #predicts, _ = load_labels(predict_file)

    ## load the gt
    #test_dataset = TSVDataset(test_data)
    #test_label_file = test_dataset.get_data('test', 'label')
    #gts, label_to_idx = load_labels(test_label_file)

    ## calculate the confusion matrix
    #confusion_pred_gt = {}
    #confusion_gt_pred = {}
    #update_confusion_matrix(predicts, gts, threshold, 
            #confusion_pred_gt, 
            #confusion_gt_pred)

    #return {'predicts': predicts, 
            #'gts': gts, 
            #'confusion_pred_gt': confusion_pred_gt, 
            #'confusion_gt_pred': confusion_gt_pred,
            #'label_to_idx': label_to_idx}

def readable_confusion_entry(entry):
    '''
    entry: dictionary, key: label, value: count
    '''
    import qd_common
    return qd_common.readable_confusion_entry(entry)
    #label_count = [(label, entry[label]) for label in entry]
    #label_count.sort(key=lambda x: -x[1])
    #total = sum([count for label, count in label_count])
    #percent = [1. * count / total for label, count in label_count]
    #cum_percent = np.cumsum(percent)
    #items = []
    #for i, (label, count) in enumerate(label_count):
        #if i >= 5:
            #continue
        #items.append((label, '{}'.format(count), '{:.1f}'.format(100. *
            #percent[i]),
            #'{:.1f}'.format(100. * cum_percent[i])))
    #return items

def get_target_images(predicts, gts, cat, threshold):
    import qd_common
    return qd_common.get_target_images(predicts, gts, cat, threshold)
    #image_aps = []
    #for key in predicts:
        #rects = predicts[key]
        #curr_gt = [g for g in gts[key] if cat == 'any' or g['class'] == cat]
        #curr_pred = [p for p in predicts[key] if cat == 'any' or (p['class'] == cat and
                #p['conf'] > threshold)]
        #if len(curr_gt) == 0 and len(curr_pred) == 0:
            #continue
        #curr_pred = sorted(curr_pred, key=lambda x: -x['conf'])
        #ap = calculate_image_ap([p['rect'] for p in curr_pred],
                #[g['rect'] for g in curr_gt])
        #image_aps.append((key, ap))
    #image_aps = sorted(image_aps, key=lambda x: x[1])
    ##image_aps = sorted(image_aps, key=lambda x: -x[1])
    #target_images = [key for key, ap in image_aps]
    #return target_images, image_aps

def gt_predict_images(predicts, gts, test_data, target_images, start_id, threshold,
        label_to_idx, image_aps, test_data_split='test'): 
    import tsv_io
    return tsv_io.gt_predict_images(predicts, gts, test_data, target_images, start_id, threshold,
        label_to_idx, image_aps, test_data_split)
    #test_dataset = TSVDataset(test_data)
    #test_tsv = TSVFile(test_dataset.get_data(test_data_split))
    #for i in xrange(start_id, len(target_images)):
        #key = target_images[i]
        #logging.info('key = {}, ap = {}'.format(key, image_aps[i][1]))
        #idx = label_to_idx[key]
        #row = test_tsv.seek(idx)
        #im = img_from_base64(row[2])
        #origin = np.copy(im)
        #im_gt = np.copy(im)
        #draw_bb(im_gt, [g['rect'] for g in gts[key]],
                #[g['class'] for g in gts[key]])
        #im_pred = im
        #rects = [p for p in predicts[key] if p['conf'] > threshold]
        #draw_bb(im_pred, [r['rect'] for r in rects],
                #[r['class'] for r in rects], 
                #[r['conf'] for r in rects])
        #yield key, origin, im_gt, im_pred, image_aps[i][1]


def categoy_check():
    data = 'office_v2.12_with_bb'
    test_data = 'office_v2.12_with_bb'
    #test_data = 'office_v2.12_no_bb'
    #cat = 'book'
    cat = 'book'
    threshold = 0.2
    expid = 'A_noreorg_burnIn5e.1_tree_initFrom.imagenet.A_bb_nobb'
    #expid = 'A_noreorg_burnIn5e.1_tree_initFrom.imagenet.A_bb_only'

    # visualize the training data
    populate_dataset_details(data)
    populate_dataset_details(test_data)
    #visualize_tsv2(data, 'train', cat)
    #visualize_tsv2(test_data, 'train', cat)

    # visualize the test data
    #visualize_tsv2(data, 'test', cat)

    # visualize the test images which has been predicted as that category
    confusion_result = get_confusion_matrix(
            data='office_v2.12', net='darknet19_448', test_data=test_data, 
            expid=expid)
    # use get_confusion_matrix
    confusion_pred_gt = confusion_result['confusion_pred_gt']
    confusion_gt_pred = confusion_result['confusion_gt_pred']
    predicts = confusion_result['predicts']
    gts = confusion_result['gts']
    label_to_idx = confusion_result['label_to_idx']
    # ---------------------------------------
    logging.info('confusion entry: predict -> gt: {}'.format(
        pformat(confusion_pred_gt[cat])))
    logging.info('confusion entry: gt -> predict: {}'.format(
        pformat(confusion_gt_pred[cat])))

    ap = calculate_ap(predicts, gts, cat)
    logging.info('re-calculate ap = {}'.format(ap))
    
    # find all images whose prediction includes the cat
    # check teh accuracy first
    target_images, image_aps = get_target_images(predicts, gts, cat, threshold)
    for key, im_gt, im_pred, ap in gt_predict_images(predicts, gts, test_data,
            target_images, 0,
            threshold, label_to_idx, image_aps):
        show_images([im_gt, im_pred], 1, 2)

def generate_pipe_dataset():
    # convert the data to a tsv format, which can be used for training
    folder = '/home/jianfw/data/raw_data/pipe'
    files = glob.glob(op.join(folder, '*'))
    image_file_names = []
    for f in files:
        if f.endswith('jpg'):
            image_file_names.append(f)
    random.seed(777)
    random.shuffle(image_file_names)
    test_image_names = image_file_names[:20]
    train_image_names = image_file_names[20:]

    def gen_rows(image_names):
        for f in image_names:
            im = cv2.imread(f)
            height, width = im.shape[:2]
            txt_file = f.replace('.jpg', '.txt')
            ann = load_list_file(txt_file)
            rects = []
            for one_ann in ann:
                an_numbers = one_ann.split(' ')
                assert len(an_numbers) == 5
                label = int(float(an_numbers[0]))
                assert label == 0
                bb_x, bb_y, bb_width, bb_height = [float(n) for n in
                    an_numbers[1:]]
                bb_x = bb_x * width
                bb_y = bb_y * height
                bb_width = bb_width * width
                bb_height = bb_height * height
                x0 = bb_x - bb_width / 2.
                y0 = bb_y - bb_height / 2.
                x1 = bb_x + bb_width / 2.
                y1 = bb_y + bb_height / 2.
                rect = {'class': 'pipe', 'rect': [x0, y0, x1, y1]}
                rects.append(rect)
            yield op.basename(f), json.dumps(rects), base64.b64encode(read_to_buffer(f))
    dataset = TSVDataset('pipe')
    logging.info('generate train...')
    tsv_writer(gen_rows(train_image_names), 
            dataset.get_data('train'))
    logging.info('generate test...')
    tsv_writer(gen_rows(test_image_names),
            dataset.get_data('test'))
    populate_dataset_details('pipe')


def convert_car_to_tsv(list_file, image_folder, annotation_folder, out_tsv):
    image_names = load_list_file(list_file)
    random.seed(777)
    random.shuffle(image_names)
    def gen_rows():
        for image_name in image_names:
            image_path = op.join(image_folder, '{}.png'.format(image_name))
            annotation_path = op.join(annotation_folder, '{}.txt'.format(image_name))
            if cv2.imread(image_path) is None:
                logging.info('skip image: {}'.format(image_name))
                continue
            annotations = load_list_file(annotation_path)
            rects = []
            for ann in annotations:
                parts = [float(p) for p in ann.split(' ')]
                assert len(parts) == 5
                assert parts[-1] == 1
                rect = {'rect': parts[:-1], 'class': 'car'}
                rects.append(rect)
            yield image_name, json.dumps(rects), \
                base64.b64encode(read_to_buffer(image_path))
    tsv_writer(gen_rows(), out_tsv)

def generate_carpk_dataset():
    rawset_root = '/mnt/sdb/data/raw_data/CARPK_PUCPR_'
    name = 'CARPK'
    data_folder = op.join(rawset_root, '{}_devkit'.format(name), 'data')
    imagesets_path = op.join(data_folder, 'ImageSets')
    splits = ['train', 'test']
    dataset = TSVDataset(name)
    for split in splits:
        list_file = op.join(imagesets_path, '{}.txt'.format(split))
        image_folder = op.join(data_folder, 'Images')
        annotation_folder = op.join(data_folder, 'Annotations')
        convert_car_to_tsv(list_file, image_folder, annotation_folder,
                dataset.get_data(split))
    populate_dataset_details(name)

def get_all_model_expid():
    import qd_common
    return qd_common.get_all_model_expid()
    #names = os.listdir('./output')
    #return names

def parse_data_net(full_expid, expid):
    data_net = full_expid.replace(expid, '')
    all_net_prefix = ['darknet', 'resnet']
    for net_prefix in all_net_prefix:
        idx = data_net.find(net_prefix)
        if idx == -1:
            continue
        # there is an underscore between data and net. Another underscore at
        # the end
        return data_net[: idx - 1], data_net[idx: -1]
    return None, None

def get_parameters_by_full_expid(full_expid):
    import qd_common
    return qd_common.get_parameters_by_full_expid(full_expid)
    #yaml_file = op.join('output', full_expid, 'parameters.yaml')
    #if not op.isfile(yaml_file):
        #return None
    #param = load_from_yaml_file(yaml_file)
    #if 'data' not in param:
        #param['data'], param['net'] = parse_data_net(full_expid,
                #param['expid'])
    #return param

def get_all_data_info():
    import tsv_io 
    return tsv_io.get_all_data_info()
    #names = os.listdir('./data')
    #name_splits_labels = []
    #names.sort(key=lambda n: n.lower())
    #for name in names:
        #dataset = TSVDataset(name)
        #if not op.isfile(dataset.get_labelmap_file()):
            #continue
        #labels = dataset.load_labelmap()
        #valid_splits = []
        #if len(dataset.get_train_tsvs()) > 0:
            #valid_splits.append('train')
        #for split in ['trainval', 'test']:
            #if not op.isfile(dataset.get_data(split)):
                #continue
            #valid_splits.append(split)
        #name_splits_labels.append((name, valid_splits, labels))
    #return name_splits_labels

def test_devonc():
    docstring
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net('./output/voc20_darknet19_A_deconvincreasedim/train.prototxt',
            './output/voc20_darknet19_A_deconvincreasedim/snapshot/model_iter_10000.caffemodel',
            caffe.TRAIN)
    import ipdb;ipdb.set_trace()

def test_dataset_op_select():
    #data = 'CocoBottle1024Merge'
    data = 'CocoBottle1024DrinkY'
    #data = 'CARPK'
    select_image = 10
    select_bb = 5
    run_dataset_op_select(data, select_image, select_bb)

    #select_bb_per_label = 1000
    #run_dataset_op_select_by_label(data, select_image, select_bb_per_label)

    #for select_image in [5, 10, 50, 100, 500]:
        #for select_bb in [5, 10, 50, 100, 500]:
            #run_dataset_op_select(select_image, select_bb)

def run_dataset_op_select_by_label(data, select_image, select_bb_per_label):
    out_data = '{}_selectbylabel.{}.{}'.format(data, select_image,
            select_bb_per_label)
    from qd_util import dataset_op_select_by_label
    dataset_op_select_by_label(data, select_image,
            select_bb_per_label, out_data)

def run_dataset_op_select(data, select_image, select_bb):
    keep_others = False
    out_data = '{}_select.{}.{}.{}'.format(data, select_image, select_bb,
            'keep' if keep_others else 'nokeep')
    dataset_op_select(data, 
            select_image=select_image, 
            select_bb=select_bb, 
            keep_others=False,
            out_data=out_data)

def yolo_old_to_new_test():
    exp_name = 'voc20_darknet19_A'
    old_weight = op.join('output', exp_name, 'snapshot',
            'model_iter_11000.caffemodel')
    old_proto = op.join('output', exp_name, 'train.prototxt')
    new_weight = \
        'output/voc20_darknet19_448_B_fullGpu_maxIter.11000/snapshot/model_iter_11000.caffemodel'
    from qd_common import yolo_old_to_new
    yolo_old_to_new(old_proto, old_weight, new_weight)

def test_dataset_op_removelabel():
    from qd_util import dataset_op_removelabel
    #dataset_op_removelabel('voc20')
    populate_dataset_details('voc20_removelabel')

def test_dataset_op_tilebb():
    from qd_util import dataset_op_tilebb
    dataset_op_tilebb('CARPK_select.5.5.nokeep')

def test_gen_honeypot_all_imagenet():
    from qd_util import load_imagenet_fname_to_url
    dataset = TSVDataset('imagenet')
    rows = tsv_reader(dataset.get_train_tsv())
    fname_to_url = load_imagenet_fname_to_url()
    cache_file = op.join('output', 'honey_pot_cache', 'a.yaml')
    if op.isfile(cache_file):
        x = load_from_yaml_file(cache_file)
        info = x['info']
        start = x['finished']
    else:
        info = []
        start = 0
    def gen_rows():
        for curr in info:
            yield curr['url'], curr['class']
    def save():
        tsv_writer(gen_rows(), '/home/jianfw/work/to_yuxiao/url_class.csv')
        write_to_yaml_file(info,
                '/home/jianfw/work/to_yuxiao/ground_truth.yaml')
    tsv = TSVFile(dataset.get_train_tsv())
    num_train = dataset.get_num_train_image()
    idx = range(num_train)
    random.shuffle(idx)
    logging.info(','.join(map(str, idx[:10])))

    for i, j in enumerate(idx):
        row = tsv.seek(j)
        if i < start:
            continue
        if (i % 20) == 0:
            logging.info('{}: {}'.format(i, len(info)))
            if len(info) > 0 or i > 0:
                write_to_yaml_file({'info': info, 'finished': i}, cache_file)
        if len(row) < 3:
            continue
        if row[0] not in fname_to_url:
            continue
        url = fname_to_url[row[0]]
        url_im = url_to_image(url)
        if url_im is None:
            continue
        local_im = img_from_base64(row[2])
        if len(row[1]) == 0:
            continue
        rects = json.loads(row[1])
        local_height, local_width = local_im.shape[:2]
        hratio = url_im.shape[0] / float(local_height)
        wratio = url_im.shape[1] / float(local_width)
        # we only select one label from rects
        random.shuffle(rects)
        select_class = rects[0]['class']
        rects = [r for r in rects if r['class'] == select_class]
        assert len(rects) >= 1
        for g in rects:
            g['rect'][0] = g['rect'][0] * wratio
            g['rect'][2] = g['rect'][2] * wratio
            g['rect'][1] = g['rect'][1] * wratio
            g['rect'][3] = g['rect'][3] * wratio
        curr = {'url': url,
                'class': select_class,
                'rects_in_url': rects,
                'height_in_url': url_im.shape[0],
                'width_in_url': url_im.shape[1]}
        info.append(curr)
        if (len(info) % 100) == 0:
            save()
    save()

def test_generate_honeypot():
    name_noffsets = [('horse', 'n02374451'),
            ('bird', 'n01503061'),
            ('bowl', 'n02881193'),
            ('dog', 'n02084071'),
            ('cup', 'n03147509'),
            ('microvave', 'n03761084'),
            ]
    select = 200


    noffset_to_name = {noffset: name for name, noffset in name_noffsets}
    name_to_noffset = {name: noffset for name, noffset in name_noffsets}

    from taxonomy import load_voc_xml

    # collect all xml
    xml_folder = '/mnt/sdb/data/raw_data/imagenet/Imagenet3kAnnotation'
    all_xml_path = []
    for name, noffset in name_noffsets:
        xmls = glob.glob(op.join(xml_folder, noffset, '*.xml'))
        all_xml_path.extend([(xml, noffset) for xml in xmls])
    random.seed(777)
    random.shuffle(all_xml_path)

    # load the pairs name, url
    name_urls = tsv_reader('/mnt/sdb/data/raw_data/imagenet/fall11_urls.txt')
    name_to_url = {}
    for x in name_urls:
        if len(x) != 2:
            logging.info('skip: {}'.format('\t'.join(x)))
            continue
        name, url = x
        name_to_url[name] = url
  
    # select the valid 200 image
    def gen_meta_data():
        for xml_path, noffset in all_xml_path:
            name = op.splitext(op.basename(xml_path))[0]
            if name in name_to_url:
                im = url_to_image(name_to_url[name])
                if im is None:
                    continue
                yield name, noffset, xml_path, name_to_url[name], im

    def calibrate(): 
        for name, noffset, xml_path, url, im in gen_meta_data():
            gts, height, width = load_voc_xml(xml_path)
            gts = [g for g in gts if g['class'] in noffset_to_name]
            for g in gts:
                g['class'] = noffset_to_name[g['class']]
            if len(gts) == 0:
                continue
            if height != im.shape[0] or width != im.shape[1]:
                logging.info('calibrate the bounding box: {}-{}, {}-{}'.format(
                    im.shape[0], height, im.shape[1], width))
                hratio = im.shape[0] / float(height)
                wratio = im.shape[1] / float(width)
                for g in gts:
                    g['rect'][0] = g['rect'][0] * wratio
                    g['rect'][2] = g['rect'][2] * wratio
                    g['rect'][1] = g['rect'][1] * wratio
                    g['rect'][3] = g['rect'][3] * wratio
            yield url, im, gts, noffset
    
    final_list = []
    c = 0
    for i, (url, im, gt, noffset) in enumerate(calibrate()):
        logging.info(i)
        gt = [g for g in gt if g['class']  == noffset]
        final_list.append((url, gt))
        if len(final_list) >= select:
            break
    
    # write to the csv for the mturk
    def gen_rows():
        for url, g in final_list:
            yield url, g[0]['class']

    def gen_url_rects():
        for url, g in final_list:
            yield url, json.dumps(g)

    tsv_writer(gen_rows(), 'url_label.csv')
    tsv_writer(gen_url_rects(), 'url_rects.csv')
    
def force_negative_visualization():
    from qd_util import draw_grid
    from qd_util import draw_gt, draw_circle
    ious = read_blob('ious.bin')
    target_obj_noobj_before = read_blob('./target_obj_noobj_before.bin')
    target_obj_noobj_after = read_blob('./target_obj_noobj_after.bin')
    data = np.load('data.npy')
    bbs = read_blob('./bbs.bin')

    num_image, num_anchor, height, width = target_obj_noobj_before.shape

    # load the image; -- image
    all_image = network_input_to_image(data, [104, 117, 123])
    # place the grid on the image;
    for b in xrange(num_image):
        logging.info(b)
        image = all_image[b]
        im_height, im_width = image.shape[:2]
        draw_grid(image, height, width)
        # place the ground truth on the image, -- gt
        blob_truth = read_blob('./blob_truth.bin')
        curr_truth = blob_truth[b]
        num_gt = draw_gt(image, curr_truth)
        if num_gt == 0:
            continue
        for x in xrange(num_gt):
            center_x = int(curr_truth[x * 5 + 0] * im_width)
            center_y = int(curr_truth[x * 5 + 1] * im_height)
            draw_circle(image, (center_x, center_y), radius=1,
                    color=[255, 0, 0])

        curr_obj_noobj_before = target_obj_noobj_before[b]
        curr_obj_noobj_after = target_obj_noobj_after[b]
        enforced = np.where(np.logical_and(curr_obj_noobj_after == 0, 
            curr_obj_noobj_before != 0))
        all_image = [None] * num_anchor
        for a in xrange(num_anchor):
            all_image[a] = copy.deepcopy(image)
        for i in xrange(len(enforced[0])):
            a = enforced[0][i]
            h = enforced[1][i]
            w = enforced[2][i]
            center_y = int((h + 0.5) / height * im_height)
            center_x = int((w + 0.5) / width * im_width)
            draw_circle(all_image[a], (center_x, center_y), radius=2,
                    color=[0,0,255])
            nx, ny, nw, nh = bbs[b, a, h, w, :]
            nx = nx * im_width
            nw = nw * im_width
            ny = ny * im_height
            nh = nh * im_height
            draw_bb(all_image[a], [[nx - 0.5 * nw, ny - 0.5 * nh, 
                nx + 0.5 * nw, ny + 0.5 * nh]], [''])

        for a in xrange(num_anchor):
            show_image(all_image[a])


        # place predicted bouding box -- bbs

        # place the anchor box in the image
        # verify the predicted box = anchor + offset -- xy, wh

            # create a dummy grid image
        #show_image(image)

def test_unavailable_image():
    #x = url_to_image('http://farm3.static.flickr.com/2083/2342713627_c048612c3b.jpg')
    x = url_to_image('http://static.flickr.com/211/473832397_e4d5d7ca86.jpg')

def test_add_prediction_into_train():
    from qd_util import load_predict
    from fast_rcnn.nms_wrapper import nms
    data = 'CARPK_select.5.5.nokeep_iter1_iter1_iter1'
    out_data = '{}_iter1'.format(data)
    #expid = 'A_dataop9450_fullGpu_randomScaleMin2_randomScaleMax4_softmaxByValid_softmaxWeight0.2_ignoreNegativeFirst_notIgnore12800'
    expid = 'A_dataop3998_fullGpu_randomScaleMin2_randomScaleMax4_initFrom_softmaxByValid_softmaxWeight0.2_ignoreNegativeFirst_notIgnore12800'
    c = CaffeWrapper(data=data,
            test_data=data,
            net='darknet19_448',
            expid=expid,
            load_parameter=True)
    m = c.best_model()
    predict_file = c._predict_file(m)
    logging.info(predict_file)
    predicts = load_predict(predict_file)
    dataset = TSVDataset(data)
    train_file = dataset.get_train_tsv()
    rows = tsv_reader(train_file)
    def gen_rows():
        for row in rows:
            iname = row[0]
            predict_bb = [b for b in predicts[iname] if b['conf'] > 0.9]
            gt_bb = json.loads(row[1])
            num_rect_before = len(gt_bb)
            for b in gt_bb:
                b['conf'] = 1
            predict_bb.extend(gt_bb)
            nms_input = np.zeros((len(predict_bb), 5), dtype=np.float32)
            for i, b in enumerate(predict_bb):
                nms_input[i, :4] = b['rect']
                nms_input[i, 4] = b['conf']
            keep = nms(nms_input, c._kwargs.get('yolo_nms', 0.45), False, device_id=0)
            rects = [predict_bb[k] for k in keep]
            num_rect_after = len(rects)
            logging.info('{}->{}'.format(num_rect_before, num_rect_after))
            yield row[0], json.dumps(rects), row[-1]

    out_dataset = TSVDataset(out_data)
    tsv_writer(gen_rows(), out_dataset.get_train_tsv())
    src_file = dataset.get_test_tsv_file()
    dest_file = out_dataset.get_test_tsv_file()
    if not op.islink(dest_file):
        os.symlink(op.relpath(src_file, op.dirname(dest_file)),
                dest_file)
    src_file = dataset.get_labelmap_file()
    dest_file = out_dataset.get_labelmap_file()
    if not op.islink(dest_file):
        os.symlink(op.relpath(src_file, op.dirname(dest_file)),
                dest_file)
    populate_dataset_details(out_data)

def incomplete_iterative():
    debug = True
    debug = False

    all_task = []
    #monitor_train_only = True
    monitor_train_only = False
    #data = 'CocoBottle1024Merge'

    #init_data = 'CocoBottle1024Merge_selectbylabel.10.1'

    #data = 'CARPK'
    data = 'CocoBottle1024DrinkY'
    data = 'CocoBottle1024DrinkY'
    num_image = 10
    num_bb = 5
    #num_image = 10000
    init_data = '{}_select.{}.{}.nokeep'.format(
            data, num_image, num_bb)

    #for k in [5, 10, 25, 50]:
        #init_data = 'CARPK_select.{}.{}.nokeep'.format(
                #5,
                #k)

    all_task.append({'data': init_data,
        'debug': debug,
        'with_background': True,
        'monitor_train_only': monitor_train_only})

    #all_p = []
    #all_task = []
    #for select_image in [5, 10, 50, 100, 500]:
        #for select_bb in [5, 10, 50, 100, 500]:
            #init_data = 'CARPK_select.{}.{}.nokeep'.format(select_image,
                    #select_bb)
            #all_task.append(init_data)
    if debug:
        incomplete_iterative_one(({}, [0]), all_task[0])
    else:
        if monitor_train_only:
            all_resouce = get_all_resources(exclude=['djx'], num_gpu=2)
        else:
            all_resouce = get_all_resources(num_gpu=4)
        b = BatchProcess(all_resouce,
                all_task,
                incomplete_iterative_one)
        b.run()

def incomplete_iterative_one(resource, task):
    # train the origin config
    data = task['data']
    init_data = data
    debug = task['debug']
    monitor_train_only = task['monitor_train_only']
    net = 'darknet19_448'
    max_iter = 10000
    #max_iter = 10
    #expid = 'R90'
    #expid = 'AnyRotate'
    if data.startswith('CAR'):
        #expid = 'MB1' # multi tsv box
        #expid = 'MB' # multi tsv box
        expid = 'MB'
    else:
        assert data.startswith('CocoBottle')
        expid = 'IB' # iterative B
    if not task['with_background']:
        expid = '{}NoBkg'.format(expid)
    yolo_nms = 0.2
    data_weight = 3
    weight_decay = 0.0005
    #weight_decay = 0.0005
    full_run_phase = [5, 9]
    yolo_not_ignore_negative_seen_images = 64 * 200
    #yolo_not_ignore_negative_seen_images = 64 * 2000
    common_param = {
        'max_iters': max_iter,
        'ovthresh': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'yolo_nms': yolo_nms,
        'yolo_not_ignore_negative_seen_images': yolo_not_ignore_negative_seen_images,
        'yolo_random_scale_max': 4,
        'yolo_random_scale_min': 2,
        'yolo_softmax_norm_by_valid': True,
        'yolo_full_gpu': True,
        'test_input_sizes': [1248],
        'ignore_negative_first_batch': True,
        'monitor_train_only': False,
        'yolo_test_maintain_ratio': True,
        'net': net,
        'detmodel': 'yolo',
    }
    tsv_box_max_samples = [[50], [1]]
    if expid.startswith('MB'):
        common_param['rotate_with_90'] = [[False, True, True, True], [False]]
        common_param['rotate_max'] = [[0, 0, 10, 180], [0]]
        common_param['box_data_param_weightss'] = [[1, 1, 1, 1], [1]]
        tsv_box_max_samples = [[50, 50, 50, 50], [1]]
    elif expid == 'R90':
        common_param['rotate_with_90'] = [[True], [False]]
        #common_param['rotate_with_90'] = [[False], [False]]
        tsv_box_max_samples = [[50], [1]]
        #tsv_box_max_samples = [[1], [1]]
    
    first_batch_objectiveness_enhancement_phase_from = 10000
    first_batch_objectiveness_enhancement = False
    first_batch_objectiveness_enhancement_weight = 1
    if first_batch_objectiveness_enhancement:
        expid = '{}FirstObjEnhance{}'.format(expid,
                first_batch_objectiveness_enhancement_weight)
        common_param['first_batch_objectiveness_enhancement'] = True
        common_param['first_batch_objectiveness_enhancement_weight'] = first_batch_objectiveness_enhancement_weight
    if not all(t == 1 for s in tsv_box_max_samples for t in s):
        common_param['tsv_box_max_samples'] = tsv_box_max_samples
        expid = '{}TsvBoxSamples{}'.format(expid, 
                '.'.join([str(t) for s in tsv_box_max_samples for t in s]))
    if data_weight != 1:
        if data.startswith('CAR'):
            expid = '{}CarWeight{}'.format(expid, data_weight)
    if data.startswith('Coco'):
        assert data_weight == 3
    base_on_voc20 = False
    if base_on_voc20:
        expid = '{}BVoc20'.format(expid)
    if max_iter != 10000:
        expid = '{}MaxIter{}'.format(expid, max_iter)
    if monitor_train_only:
        common_param['monitor_train_only'] = True
    if weight_decay != 0.0005:
        common_param['weight_decay'] = weight_decay
        expid = '{}Decay{}'.format(expid, weight_decay)
    if yolo_not_ignore_negative_seen_images != 12800:
        expid = '{}NotIgnore{}'.format(
                expid,
                yolo_not_ignore_negative_seen_images)
    common_param['expid'] = expid
    logging.info(pformat(common_param))

    for i in xrange(10):
        if task['with_background']:
            dataset_ops = [{'op':'remove'},
                           {'op':'add',
                                 'name': data,
                                 'source':'train',
                                 'weight': data_weight},
                           {'op': 'add',
                                 'name': 'voc20_removelabel',
                                 'source': 'train',
                                 'weight': 1},
                                ]
        else:
            dataset_ops = []
        param = copy.deepcopy(common_param)
        if data.startswith('CAR'):
            param['test_data'] = 'CARPK'
        elif data.startswith('CocoBottle1024Merge'):
            param['test_data'] = 'CocoBottle1024Merge'
        elif data.startswith('CocoBottle1024Drink'):
            param['test_data'] = 'CocoBottle1024DrinkYW2'
        else:
            assert False
        param['data'] = data
        param['dataset_ops'] = dataset_ops
        if i != 0:
            param['basemodel'] = op.join('output',
                    '{}_{}_{}'.format(previous_data, net, expid),
                    'snapshot',
                    'model_iter_{}.caffemodel'.format(max_iter))
        if i == first_batch_objectiveness_enhancement_phase_from:
            expid = '{}FirstObjEnhance{}'.format(expid,
                    first_batch_objectiveness_enhancement_weight)
        if i >= first_batch_objectiveness_enhancement_phase_from:
            param['first_batch_objectiveness_enhancement'] = True
            param['first_batch_objectiveness_enhancement_weight'] = first_batch_objectiveness_enhancement_weight
            param['expid'] = expid

        if i in full_run_phase:
            full_param = copy.deepcopy(param)
            full_param['ignore_negative_first_batch'] = False
            full_param['yolo_not_ignore_negative_seen_images'] = 0
            full_param['expid'] = '{}FullRun'.format(expid)
            task_processor(resource, full_param)
        #if base_on_voc20:
            #param['basemodel'] = op.join('output',
                    #'_'.join(['voc20', net, 'B']),
                    #'snapshot',
                    #'model_iter_10022.caffemodel')
        
        if debug:
            logging.info(pformat(param))
            return
        task_processor(resource, param)
        #b = BatchProcess(get_all_resources(), [param], task_processor)
        #b._availability_check = False
        #b.run()
        param['test_data'] = data
        task_processor(resource, param)

        #b = BatchProcess(get_all_resources(), [param], task_processor)
        #b._availability_check = False
        #b.run()

        from qd_util import add_prediction_into_train
        previous_data = data
        c = CaffeWrapper(data=data,
                test_data=data,
                net=net,
                expid=expid,
                load_parameter=True)
        m = c.best_model()
        predict_file = c._predict_file(m)
        if i >= first_batch_objectiveness_enhancement_phase_from:
            out_data = '{}_{}{}_NMS{}From{}'.format(init_data, 
                    expid, 
                    i + 1,
                    yolo_nms, 
                    first_batch_objectiveness_enhancement_phase_from)
        else:
            out_data = '{}_{}{}_NMS{}'.format(init_data, 
                    expid, 
                    i + 1,
                    yolo_nms)
        data = add_prediction_into_train(data, 
                predict_file, 
                yolo_nms,
                out_data)

def replace_empty_by_zerolist(data, out_data):
    splits = ['train', 'test']
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)
    for split in splits:
        if split == 'train':
            tsv = TSVFile(dataset.get_data(split))
            total = dataset.get_num_train_image()
            idx = range(total)
            random.shuffle(idx)
            logging.info(','.join(map(str, idx[:10])))
            def gen_train_rows():
                c = 0
                t = 0
                for i in idx:
                    row = tsv.seek(i)
                    if img_from_base64(row[2]) is None:
                        continue
                    if row[1] == '':
                        c = c + 1
                        if (c % 100) == 0:
                            logging.info('{}-{}-{}'.format(c, t, 1. * c / t))
                        row[1] = json.dumps([])
                    t = t + 1
                    yield row
                logging.info('{}-{}-{}'.format(c, total, 1. * c / total))
            tsv_writer(gen_train_rows(), out_dataset.get_data(split))
        else:
            rows = tsv_reader(dataset.get_data(split))
            def gen_rows():
                c = 0
                t = 0
                for row in rows:
                    if row[1] == '':
                        c = c + 1
                        row[1] = json.dumps([])
                    if img_from_base64(row[2]) is None:
                        continue
                    t = t + 1
                    yield row
                logging.info('{}-{}-{}'.format(c, t, 1. * c / t))
            tsv_writer(gen_rows(), out_dataset.get_data(split))
    shutil.copyfile(dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file())
    populate_dataset_details(out_data)

def extract_dataset(data):
    dataset = TSVDataset(data)
    from process_tsv import visualize_tsv
    tsv = '/vigssd/imagenet2012/tsv480/train.resize480.shuffled.tsv'
    visualize_tsv(tsv, 
            tsv_label=tsv, 
            out_folder='/mnt/sda/data/images/{}/train'.format(data), 
            label_idx=1)

def parse_mturk_rects(bbs_str):
    bbs = json.loads(bbs_str)
    r = False
    result = []
    for bb in bbs:
        x0, y0 = bb['left'], bb['top']
        x1, y1 = (x0 + bb['width']), (y0 + bb['height'])
        rect = {'rect': [x0, y0, x1, y1], 'class': bb['label']}
        result.append(rect)
    return result

def test_check_mturk_result():
    import pandas
    #csv_file = '/home/jianfw/work/share/Batch_3091978_batch_results.csv'
    csv_file = '/home/jianfw/work/share/Batch_3104588_batch_results.csv'
    honey_pot_file = '/home/jianfw/work/to_yuxiao/ground_truth.yaml'
    
    info = {}
    logging.info('loading honey pot')
    hps = load_from_yaml_file(honey_pot_file)
    url_to_hp = {hp['url']: hp for hp in hps}
    
    logging.info('loading mt result')
    mt_result = pandas.read_csv(csv_file)
    info['total_assignment'] = len(mt_result)

    mt_result['is_honey'] = mt_result.apply(lambda row: row['Input.image_url']
            in url_to_hp, axis=1)
    logging.info('number of honey pot {}'.format(mt_result['is_honey'].sum()))

    def is_bb_correct(row):
        url = row['Input.image_url']
        hp = url_to_hp[url]
        bbs = json.loads(row['Answer.annotation_data'])
        r = False
        for bb in bbs:
            x0, y0 = bb['left'], bb['top']
            x1, y1 = (x0 + bb['width']), (y0 + bb['height'])
            for gt in hp['rects_in_url']:
                iou = calculate_iou(gt['rect'], [x0, y0, x1, y1])
                if iou > 0.3:
                    r = True
                    break
        return r
    mt_result['is_honey_correct'] = mt_result.apply(lambda row: row['is_honey'] and is_bb_correct(row),
            axis=1)

    mt_worker = mt_result.groupby('WorkerId').agg({'is_honey': ['size', 'sum'],
        'is_honey_correct': ['size', 'sum']})
    mt_worker['honeytask_acc'] = mt_worker.apply(lambda row:
            (row[('is_honey_correct', 'sum')] / row[('is_honey', 'sum')]) if row[('is_honey', 'sum')]
            else -1, axis=1)
    info['total_worker'] = len(mt_worker)
    mt_worker_mean = mt_worker.mean()
    mt_worker_std = mt_worker.std()
    info['task_per_worker_mean'] =  mt_worker_mean[0]
    info['task_per_worker_std'] = mt_worker_std[0]
    info['honeytask_acc'] = 1. * sum(mt_result['is_honey_correct']) / \
        sum(mt_result['is_honey'])
    
    bad_workers = mt_worker[(mt_worker['honeytask_acc'] >= 0) &
            (mt_worker['honeytask_acc'] < 0.5)]
    good_workers = mt_worker[mt_worker['honeytask_acc'] >= 0.5]
    qualityunkonw_workers = mt_worker[mt_worker['honeytask_acc'] < 0]

    assert len(bad_workers) + len(good_workers) + len(qualityunkonw_workers) \
            == len(mt_worker)

    info['bad_workers'] = bad_workers

    info['num_bad_workers'] = len(bad_workers)
    info['num_good_workers'] = len(good_workers)
    info['num_qualityunknown_workers'] = len(qualityunkonw_workers)
    
    info['num_tasks_by_bad_workers'] = bad_workers[('is_honey', 'size')].sum()
    info['num_tasks_by_good_workers'] = good_workers[('is_honey',
        'size')].sum()
    info['num_tasks_by_qualityunknown_workers'] = qualityunkonw_workers[('is_honey',
        'size')].sum()
    assert info['num_tasks_by_bad_workers'] + \
            info['num_tasks_by_good_workers'] + \
            info['num_tasks_by_qualityunknown_workers'] == \
            len(mt_result)

    logging.info(pformat(info))

    mt_result['rects'] = mt_result.apply(lambda row: parse_mturk_rects(row['Answer.annotation_data']),
            axis=1)

    mt_worker_plain = mt_worker.reset_index()[['WorkerId',
        'honeytask_acc']]
    all_task = []
    for _, task in mt_result.iterrows():
        all_task.append((task['Input.image_url'],
            task['WorkerId'],
            task['rects']))
    from qd_common import list_to_dict
    from qd_common import dict_to_list
    url_to_worker_result = list_to_dict(all_task, 0)
    worker_to_consistency = {}

    #for url in url_to_worker_result:
        #all_worker_result = url_to_worker_result[url]
        #consistency_all = True
        #if len(all_worker_result) == 3:
            #consistency_all = False
        #else:
            #for i, (wi, ri) in enumerate(all_worker_result):
                #for j, (wj, rj) in enumerate(all_worker_result):
                    #if i != j:
                        #continue
                    #if not is_consistent(ri, rj):
                        #consistency_all = False
                        #break
        #if consistency_all:
            #for w, r in all_worker_result:
                #pass
                ##worker_to_consistency[w]++
            #break
        
    for url in url_to_worker_result:
        all_worker_result = url_to_worker_result[url]
        im = url_to_image(url)
        if im is None:
            continue
        all_im = []
        for worker, rects in all_worker_result:
            curr_im = im.copy()
            draw_bb(curr_im, [r['rect'] for r in rects],
                    [r['class'] for r in rects])
            all_im.append(curr_im)
        show_images(all_im, 1, len(all_im))
    
    def show_image_by_worker_id(workerId):
        tasks = mt_result[mt_result['WorkerId'] == workerId]
        for _, task in tasks.iterrows():
            url = task['Input.image_url']
            bbs_str = task['Answer.annotation_data']
            label = task['Input.objects_to_find']
            im = url_to_image(url)
            if im is None:
                logging.info('image is not available: {}'.format(
                    url))
                continue
            bbs = json.loads(bbs_str)
            r = False
            all_bb = []
            all_label = []
            for bb in bbs:
                x0, y0 = bb['left'], bb['top']
                x1, y1 = (x0 + bb['width']), (y0 + bb['height'])
                all_bb.append([x0, y0, x1, y1])
                all_label.append(bb['label'])
            draw_bb(im, all_bb, all_label)
            logging.info('request label: {}'.format(label))
            show_image(im)

    import ipdb;ipdb.set_trace() 
    for workerId in bad_workers.reset_index()['WorkerId']:
        show_image_by_worker_id(workerId)

def exclude_partial_labeled():
    csv_all = ''
    csv_mt_result = '/home/jianfw/work/share/Batch_3097725_batch_results.csv'
    import pandas as pd
    mt_result = pd.read_csv(csv_mt_result)
    import ipdb;ipdb.set_trace()
    finished = mt_result.grouopby('Input.image_url').size().reset_index()
    finished = finished[finished[0] >=2]
    
    import ipdb;ipdb.set_trace()

def count_car():
    dataset = TSVDataset('CARPK')
    for split in ['train', 'test']:
        x = dataset.get_data(split, 'label')
        rows = tsv_reader(x)
        x = 0
        c = 0
        for row in rows:
            x = x + len(json.loads(row[1]))
            c = c + 1
        logging.info(1.0 * x / c)
        logging.info(c)

def pr_curve():
    precision = []
    for line in load_list_file('a.txt'):
        if line.endswith(','):
            precision.append(float(line[:-1]))
    recall = []
    for line in load_list_file('b.txt'):
        if line.endswith(','):
            recall.append(float(line[:-1]))
    plt.figure()
    plt.plot(recall, precision)
    plt.grid()
    plt.show()
    #plt.plot(reversed(precision), 
            #reversed(recall))
    #plt.show()

def create_ssd_tsv_dataset():
    data = 'voc0712_ssd'
    out_dataset = TSVDataset(data)
    ssd_lmdb = '/home/jianfw/code/ssd/caffe/examples/VOC0712/VOC0712_trainval_lmdb'
    ssdlmdb_to_tsv(ssd_lmdb, out_dataset.get_data('train'))
    ssd_lmdb = '/home/jianfw/code/ssd/caffe/examples/VOC0712/VOC0712_test_lmdb'
    ssdlmdb_to_tsv(ssd_lmdb, out_dataset.get_data('test'))
    populate_dataset_details(data)

def ssdlmdb_to_tsv(ssd_lmdb, tsv_file):
    rows = lmdb_reader(ssd_lmdb)
    datum = caffe_pb2.AnnotatedDatum()
    def gen_rows():
        for i, (key, value) in enumerate(rows):
            datum = caffe_pb2.AnnotatedDatum()
            datum.ParseFromString(value)
            nparr = np.frombuffer(datum.datum.data, np.uint8)
            im = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
            rects = []
            for g in datum.annotation_group:
                label_idx = g.group_label - 1
                for a in g.annotation:
                    abox = [a.bbox.xmin * im.shape[1],
                            a.bbox.ymin * im.shape[0],
                            a.bbox.xmax * im.shape[1],
                            a.bbox.ymax * im.shape[0]]
                    rect = {'class': str(label_idx), 
                            'rect': abox,
                            'diff': 1 if a.bbox.difficult else 0}
                    rects.append(rect)
            yield key, json.dumps(rects), base64.b64encode(datum.datum.data)
    tsv_file_tmp = tsv_file + '.tmp.tsv'
    tsv_writer(gen_rows(), tsv_file_tmp)
    tsv = TSVFile(tsv_file_tmp)
    def gen_rows2():
        num_rows = tsv.num_rows()
        for i in xrange(num_rows):
            j = ((i - 1 + num_rows) % num_rows)
            yield tsv.seek(j)
    tsv_writer(gen_rows2(), tsv_file)

def parse_annoated_datum(value):
    datum = caffe_pb2.AnnotatedDatum()
    datum.ParseFromString(value)
    nparr = np.frombuffer(datum.datum.data, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
    rects = []
    for g in datum.annotation_group:
        label_idx = g.group_label - 1
        for a in g.annotation:
            abox = [a.bbox.xmin * im.shape[1] - 1,
                    a.bbox.ymin * im.shape[0] - 1,
                    a.bbox.xmax * im.shape[1] - 1,
                    a.bbox.ymax * im.shape[0] - 1]
            rect = {'class': str(label_idx), 
                    'rect': abox}
            rects.append(rect)
    return rects, im

def read_lmdb():
    ssd_lmdb = '/home/jianfw/code/ssd/caffe/examples/VOC0712/VOC0712_trainval_lmdb'
    dataset = TSVDataset('voc0712_ssd')
    rows_tsv = tsv_reader(dataset.get_train_tsv())
    rows = lmdb_reader(ssd_lmdb)
    for (key, value), (row_tsv) in izip(rows, rows_tsv):
        rects, im = parse_annoated_datum(value)
        rects_tsv = json.loads(row_tsv[1])
        im_tsv = img_from_base64(row_tsv[-1])
        import ipdb;ipdb.set_trace()

def parity_check():
    ssd_data = read_blob('/home/jianfw/work/ssd.data.b')
    qd_data = read_blob('/home/jianfw/work/qd.ssd.data.b')
    d = np.sum(np.abs(ssd_data[:] - qd_data[:]))
    logging.info(d)
    all_im = network_input_to_image(qd_data, 
        [104.0,
        117.0,
        123.0])
    #for im in all_im:
        #show_image(im)

    ssd_label = read_blob('/home/jianfw/work/ssd.label.b')
    qd_label = read_blob('/home/jianfw/work/qd.ssd.label.b')
    ssd_label = ssd_label.squeeze()
    qd_label = qd_label.squeeze()
    d = np.abs(ssd_label - qd_label)
    logging.info(np.sum(d[:, 1]))
    logging.info(np.sum(d[:, 0]))
    logging.info(np.sum(d[:, 2:][:]))

    import ipdb;ipdb.set_trace()


def test_ssd():
    #test_ssd_dataset()
    create_ssd_tsv_dataset()
    #parity_check()
    #read_lmdb()
    #dataset = TSVDataset('voc0712_ssd')
    #rows = tsv_reader(dataset.get_train_tsv())
    #for row in rows:
        #x = json.loads(row[1])
        #import ipdb;ipdb.set_trace()
    return

    label = read_blob("src/CCSCaffe/t.label.b")
    data = read_blob("src/CCSCaffe/t.data.b")
    all_im = network_input_to_image(data, 
        [104.0,
        117.0,
        123.0])
    label = label.squeeze()
    for l in label:
        image_id = int(l[0])
        curr_im = all_im[image_id]
        height, width = curr_im.shape[:2]
        xmin, ymin, xmax, ymax = l[3:7]
        xmin = xmin * width
        xmax = xmax * width
        ymin = ymin * height
        ymax = ymax * height
        draw_bb(curr_im, [[xmin, ymin, xmax, ymax]], [str(l[1])])

    for im in all_im:
        show_image(im)

#def test_create_rotate_training_set():
    #from qd_util import rotate_image
    #data = 'CARPK_select.5.5.nokeep'
    #max_rotate = 180
    #out_data = '{}.RRotate{}'.format(data,
            #max_rotate)
    #dataset = TSVDataset(data)
    #out_dataset = TSVDataset(out_data)

    #tsv_file = dataset.get_train_tsv()
    #rows = tsv_reader(tsv_file)

    #for row in rows:
        #im = img_from_base64(row[-1])
        #rects = json.loads(row[1])
        #angle = (2 * random.random() - 1) * max_rotate
        #angle = 30
        #rotated_image, rotated_rects = rotate_image(im, rects, angle)
        #draw_bb(rotated_image, [r['rect'] for r in rotated_rects],
                #[r['class'] for r in rotated_rects])
        #show_image(rotated_image)

def test_ssd_dataset():
    import lmdb
    from caffe.proto import caffe_pb2
    lmdb_file ='/home/jianfw/code/ssd/caffe/examples/VOC0712/VOC0712_trainval_lmdb'
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.AnnotatedDatum()

    dataset = TSVDataset('voc0712')
    out_dataset = TSVDataset('voc0712_ssd')
    rows = tsv_reader(dataset.get_data('train', 'label'))
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        assert key not in key_to_idx
        key_to_idx[key] = i
    tsv = TSVFile(dataset.get_data('train'))
    labelmap = dataset.load_labelmap()
    
    max_height = 0
    max_width = 0
    total_labels = 0
    total_labels2 = 0
    label_idx_to_label = {}
    total_diff = 0
    total_diff_size = 0
    num_difficult_lmdb = 0
    for i, (key, value) in enumerate(lmdb_cursor):
        key = op.splitext(op.basename(key))[0]
        datum.ParseFromString(value)
        nparr = np.frombuffer(datum.datum.data, np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        num_labels = 0
        idx = key_to_idx[key]
        row = tsv.seek(idx)
        rects = json.loads(row[1])
        im2 = img_from_base64(row[-1])
        d = np.sum(np.abs(np.asarray(im, np.float32) - np.asarray(im2,
            np.float32)))
        total_diff = total_diff + 1. * d / np.sum(im)
        total_diff_size = total_diff_size + 1
        for g in datum.annotation_group:
            label_idx = g.group_label - 1
            num_labels = num_labels + len(g.annotation)
            for a in g.annotation:
                abox = [a.bbox.xmin * im.shape[1] - 1,
                        a.bbox.ymin * im.shape[0] - 1,
                        a.bbox.xmax * im.shape[1] - 1,
                        a.bbox.ymax * im.shape[0] - 1]
                num_difficult_lmdb = num_difficult_lmdb + a.bbox.difficult
                def find_nn(rects, abox):
                    best_rect = None
                    best_dist = 9999
                    for rect in rects:
                        d = np.sum(np.abs(np.asarray(abox) -
                            np.asarray(rect['rect'])))
                        if d < best_dist:
                            best_rect = rect
                            best_dist = d
                    return best_rect
                rect = find_nn(rects, abox)
                if label_idx in label_idx_to_label:
                    assert rect['class'] == label_idx_to_label[label_idx]
                else:
                    label_idx_to_label[label_idx] = rect['class']
                diff = np.sum(np.abs(np.asarray(abox) -
                    np.asarray(rect['rect'])))
                assert diff < 0.0001


        num_labels2 = len(rects)
        total_labels = total_labels + num_labels
        total_labels2 = total_labels2 + num_labels2
        assert im2.shape[0] == im.shape[0]
        assert im2.shape[1] == im.shape[1]
        if (i % 500) == 0:
            logging.info(total_labels)
            logging.info(total_labels2)
            logging.info('{}'.format(1. * total_diff / total_diff_size))
            logging.info('difficult: {}'.format(num_difficult_lmdb))
    logging.info(total_labels)
    logging.info(total_labels2)
    logging.info('difficult: {}'.format(num_difficult_lmdb))

def analyze():
    data = 'Tax700V2'
    expid = 'B_noreorg'
    data_with_bb = data + '_with_bb'
    data_no_bb = data + '_no_bb'
    report = ['\n']
    r = analyze_test(data, data_with_bb,
            data_no_bb, 
            test_data=data_with_bb)
    report.extend(r)
    r = analyze_test(data, data_with_bb, data_no_bb, test_data=data_no_bb)
    report.extend(r)
    logging.info('\n'.join(report))
    import ipdb;ipdb.set_trace(context=15)

def analyze_test(data, data_with_bb, data_no_bb, test_data):
    expid = 'B_noreorg_extraconv2'
    all_expid = [expid + '_bb_only', expid + '_bb_nobb']
    datas = [data_with_bb, data_no_bb]
    report = []
    report.append('Test Data = {}'.format(test_data))
    all_caffe_task = []
    for expid in all_expid:
        c = CaffeWrapper(data=data, 
                net='darknet19_448', 
                detmodel='yolo',
                expid=expid, 
                load_parameter=True,
                test_data=test_data)
        all_caffe_task.append(c)

    all_evaluate = []
    for c in all_caffe_task:
        m = c.best_model()
        evaluate_file = c._perf_file(m)
        result = json.loads(read_to_buffer(evaluate_file))
        all_evaluate.append(result)
    
    iou_th = '0.3' if '0.3' in result['overall'] else '-1'
    all_map = [result['overall'][iou_th]['map'] for result in all_evaluate]
    all_class_aps = []
    for result in all_evaluate:
        class_ap = result['overall'][iou_th]['class_ap']
        all_class_aps.append(class_ap)

    all_dataset = [TSVDataset(data) for data in datas]
    inverted = {}
    for data, dataset in izip(datas, all_dataset):
        inverted[data] = {}
        for split in ['train', 'test']:
            inverted[data][split] = dataset.load_inverted_label(split)
    
    for expid, mAP in izip(all_expid, all_map):
        report.append('overall mAP = {}; expid = {}'.format(mAP, expid))

    label_aps = [(key, class_ap[key]) for key in class_ap]
    label_aps.sort(key=lambda x: x[1])
    for label, _ in label_aps[:10]:
        c = []
        c.append('label = {}'.format(label))
        for expid, class_ap in izip(all_expid, all_class_aps):
            c.append('AP({}) = {}'.format(
                expid[-5:],
                class_ap[label]))
        for data in datas:
            for split in ['train', 'test']:
                num = len(inverted[data][split].get(label, []))
                c.append('#{}-{} = {}'.format(data, split, num))
        report.append('; '.join(c))

    return report

def remove_invalid_box():
    data = 'Visual_Genome'
    out_data = 'VisualGenomeClean'
    source_dataset = TSVDataset(data)
    dest_dataset = TSVDataset(out_data)
    def gen_rows():
        num_has_invalid = 0
        c = 0
        for i, (key, label_str, image_str) in enumerate(source_dataset.iter_data(
            'trainval')):
            rects = json.loads(label_str)
            im = img_from_base64(image_str)
            height, width = im.shape[:2]
            to_removed = []
            invalid = False
            for rect in rects:
                r = rect['rect']
                cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                rw, rh = r[2] - r[0], r[3] - r[1]
                if cx < 0 or cx >= width \
                        or cy < 0 or cy >= height \
                        or rw < 1 or rh < 1:
                    invalid = True
                    c = c + 1
                    break
            if (i % 1000) == 0:
                logging.info(c)
            if not invalid:
                yield key, json.dumps(rects), image_str
        logging.info(str(num_has_invalid))

    tsv_writer(gen_rows(), dest_dataset.get_data('trainval'))



def check_dataset():
    dataset = TSVDataset('Tax1300SGV1_3_with_bb')
    k = 0
    for key, label_str, im_str in dataset.iter_data('train'):
        im = img_from_base64(im_str)
        logging.info(k)
        logging.info(','.join(map(str, im.shape)))
        rects = json.loads(label_str)
        for rect in rects:
            logging.info(', '.join(map(str, rect['rect'])))
            r = rect['rect']
            assert r[2] - r[0] > 1 and r[3] - r[1] > 1
            assert (r[0] + r[2]) / 2. / im.shape[1] >= 0
            assert (r[0] + r[2]) / 2. / im.shape[1] < 1
            assert (r[1] + r[3]) / 2. / im.shape[0] < 1
            assert (r[1] + r[3]) / 2. / im.shape[0] >= 0
        k = k + 1
        if k > 100:
            break
        
def torwards_10K():
    #from process_tsv import resize_dataset
    #from process_tsv import test_visualize
    #from process_tsv import gen_coco_noffset_map
    #from qd_util import create_imagenet3k_tsv
    #from qd_util import create_imagenet22k_tsv
    #from process_tsv import standarize_crawled
    #from process_tsv import populate_all_dataset_details
    #from process_tsv import clean_dataset
    from qd_util import convertcomposite_to_standard
    from qd_util import destroy_label_field
    from qd_util import merge_prediction_to_gt
    from qd_util import test_merge_prediction_to_gt
    from qd_util import gen_html_label_mapping
    from qd_util import create_voc_person
    from qd_util import distribute_dataset
    from qd_util import philly_remove
    from qd_util import merge_similar_boundingbox

    #write_to_file('\n'.join(map(str, range(1000))), 'data/imagenet2012/labelmap.txt')
    #test_merge_prediction_to_gt()
    #data = 'MSLogoClean'
    #merge_similar_boundingbox('4000_Full_setClean')
    #merge_similar_boundingbox(data)
    #populate_dataset_details(data)
    # ------------- data preparation for 10k detector
    #test_generate_honeypot()
    #test_gen_honeypot_all_imagenet()
    #test_unavailable_image()
    #test_check_mturk_result()
    #exclude_partial_labeled()
    #officev2_1()
    #officev2_11()
    #populate_all_dataset_details()
    #from process_tsv import normalize_str_in_rects
    #normalize_str_in_rects('VisualGenomeClean', 
            #'VisualGenomeClean2')
    #data = 'Tax700V3_1'
    #distribute_dataset('brand1048')
    #per_class_check()

    #dataset = TSVDataset('open_images')
    #keys = dataset.load_keys('trainval')
    #for k, key in enumerate(keys):
        #if 'd1071c2b3991d2bc' in key:
            #found = k
            #tsv = TSVFile(dataset.get_data('trainval'))
            #row = tsv.seek(k)
            #import ipdb;ipdb.set_trace(context=15)

    #for s in datas:
        #try:
            #populate_dataset_details(s)
        #except:
            #clean_dataset(s, s + 'Clean')
            #populate_dataset_details(s + 'Clean')
    #populate_dataset_details('Tax1300SGV1_1_no_bb')
    #populate_dataset_details('Tax1300SGV1_1_with_bb')
    #populate_dataset_details('imagenet3k_448')
    #data = 'imagenet1kLoc'
    #clean_dataset(data, data + 'Clean')
    #populate_dataset_details('Tax4k_V1_6_with_bb')
    #populate_dataset_details('imagenet3k_448Clean')
    #gen_html_label_mapping('TaxPerson_V1_2')
    #data = 'TaxPerson_V1_3'
    #populate_dataset_details(data + '_with_bb')
    #populate_dataset_details(data + '_no_bb')

    datas=[
        'coco2017',
        'voc0712', 
        'brand1048Clean',
        'imagenet3k_448Clean',
        'imagenet22k_448',
        'imagenet1kLocClean',
        'mturk700_url_as_keyClean',
        'crawl_office_v1',
        'crawl_office_v2',
        'Materialist',
        'VisualGenomeClean',
        'Naturalist',
        '4000_Full_setClean',
        'MSLogoClean',
        'clothingClean',
        #'open_images_clean_1',
        'open_images_clean_2',
        'open_images_clean_3',
        ]
    for data in datas:
        populate_dataset_details(data)

    #build_taxonomy_impl('./aux_data/taxonomy10k/PersonGender/PersonGenderV1',
            #data='TaxPersonGender_V1_1',
            #datas=datas, max_image_per_label=10000)
    #build_taxonomy_impl('./aux_data/taxonomy10k/PersonAge/PersonAgeV1/',
            #data='TaxPersonAge_V1_1',
            #datas=datas, 
            #max_image_per_label=10000)
    #build_taxonomy_impl('./aux_data/taxonomy10k/PersonAgeGender/PersonAgeGenderV1/',
            #data='TaxPersonAgeGender_V1_1',
            #datas=datas, 
            #max_image_per_label=10000)

    #for data in datas:
        #dataset = TSVDataset(data)
        #for split in ['train', 'test', 'trainval']:
            #if op.isfile(dataset.get_data(split)):
                #yolo_master(test_data=data, test_split=split)
    #for data in datas:
        ##philly_upload_dir('data/{}'.format(data), 
                ##'jianfw/data/qd_data',
                ##vc='input',
                ##cluster='eu2')
        #philly_upload_dir('data/{}'.format(data), 
                #'jianfw/data/qd_data',
                #vc='resrchvc',
                #cluster='gcr')
    #datas=['coco2017',
        #'voc0712', 
        #'imagenet3k_448Clean',
        #'imagenet1kLocClean',
        #'crawl_office_v1',
        #'crawl_office_v2',
        #'VisualGenomeClean',
        #'open_images_clean_1',
        #'open_images_clean_2',
        #'open_images_clean_3',
        #]
    #build_taxonomy_impl('/mnt/jianfw_desk/X',
            #data='Tax50_V1_110',
            #datas=datas,
                #max_image_per_label=110)
    #data_from = 'TaxPerson_V1_2_with_bb'
    #data_to = '{}_S'.format(data_from)
    #convertcomposite_to_standard(data_from, 
            #data_to, 
            #'train')
    #destroy_label_field(data_to, 'train')

    #populate_dataset_details(data_to)

    #populate_dataset_details('TaxPerson_V1_2_with_bb_S')
    #populate_dataset_details('TaxPerson_V1_2_with_bb_S_M1')

    #logging.info(TSVDataset('TaxPersonDetector_V1_1_debug_no_bb').get_num_train_image())
    #param = load_from_yaml_file('data/TaxPerson_V1_2/generate_parameters.yaml')
    #input_folder = param[0]
    #kwargs = param[1]
    #kwargs['data'] = 'rTaxPerson_V1_2'
    #build_taxonomy_impl(input_folder, **kwargs)

    #populate_dataset_details('TaxPerson_V1_2_S_C_with_bb')
    #populate_dataset_details('TaxPerson_V1_2_S_M1_C_with_bb')
    #create_voc_person()
    #populate_dataset_details('voc20_person')
    
    #fname = \
        #'/mnt/glusterfs/jianfw/work/qd_output/TaxPerson_V1_2_S_C_with_bb_darknet19_448_B_noreorg_extraconv2_tree/snapshot/model_iter_56684.caffemodel.TaxPerson_V1_2_with_bb.test.maintainRatio.OutTreePath.ClsIndependentNMS.report.class_ap.json'
    #fname = \
        #'/mnt/glusterfs/jianfw/work/qd_output/TaxPerson_V1_2_S_M1_C_with_bb_darknet19_448_B_noreorg_extraconv2_tree/snapshot/model_iter_67798.caffemodel.TaxPerson_V1_2_with_bb.test.maintainRatio.OutTreePath.ClsIndependentNMS.report.class_ap.json'
    #class_ap = json.loads(read_to_buffer(fname))
    #write_to_file(json.dumps(class_ap, indent=4, sort_keys=True), fname)

    #populate_dataset_details('Tax50_bb')
    #from pathos.multiprocessing import ProcessingPool as Pool
    #pool = Pool()
    #def upload_data(data):
        #philly_upload_dir('data/{}'.format(data), 
                #'jianfw/data/qd_data')
    #pool.map(upload_data, datas)
    #for data in datas:
        #philly_upload_dir('data/{}'.format(data), 
                #'jianfw/data/qd_data')
    #populate_dataset_details('TaxPerson_no_bb')

    #build_taxonomy_impl('data/TaxPerson_V1_2/taxonomy_folder',
            #data='TaxPerson_V1_2_S_C_Debug',
            #datas=['TaxPerson_V1_2_with_bb_S'],
                #max_image_per_label=10000,
                #num_test=0)
    #build_taxonomy_impl('data/TaxPerson_V1_2/taxonomy_folder',
            #data='TaxPerson_V1_2_S_M1_C',
            #datas=['TaxPerson_V1_2_with_bb_S_M1'],
                #max_image_per_label=10000,
                #num_test=0)
    #build_taxonomy_impl('./data/TaxPerson_V1_2/taxonomy_folder/',
            #data='TaxPerson_V1_3_debug',
            #datas=['coco2017'],
                #max_image_per_label=10000)
    #build_taxonomy_impl('/mnt/jianfw_desk/Person',
            #data='TaxPerson_V1',
            #datas=['coco2017',
                #'voc0712', 
                #'brand1048Clean',
                #'imagenet3k_448Clean',
                #'imagenet22k_448',
                #'imagenet1kLocClean',
                #'mturk700_url_as_keyClean',
                #'crawl_office_v1',
                #'crawl_office_v2',
                #'Materialist',
                #'VisualGenomeClean',
                #'Naturalist',
                #'4000_Full_setClean',
                #'MSLogoClean',
                #'clothingClean',
                #'open_images_clean_1',
                #'open_images_clean_2',
                #'open_images_clean_3',
                #],
            #max_image_per_label=1000)
            #num_test=20)
    #build_taxonomy_impl('./aux_data/taxonomy10k/Tax4K/Tax4KV1',
            #data='Tax4k_V1_9',
            #datas=['coco2017',
                #'voc0712', 
                #'brand1048Clean',
                #'imagenet3k_448Clean',
                #'imagenet22k_448',
                #'imagenet1kLocClean',
                #'mturk700_url_as_keyClean',
                #'crawl_office_v1',
                #'crawl_office_v2',
                #'Materialist',
                #'VisualGenomeClean',
                #'Naturalist',
                #'4000_Full_setClean',
                #'MSLogoClean',
                #'clothingClean',
                #'open_images_clean_1',
                #'open_images_clean_2',
                #'open_images_clean_3',
                #],
            #num_test=20)
    #build_taxonomy_impl('./aux_data/',
            #data='Tax700V3_1',
            #datas=['coco2017', 
                #'voc0712', 
                #'brand1048',
                #'imagenet3k_448', 
                #'imagenet22k_448', 
                #'imagenet1kLoc',
                #'crawl_office_v1', 
                #'crawl_office_v2',
                #'mturk700_url_as_key'])
    #build_taxonomy_impl('./aux_data/taxonomy10k/Tax1300/Tax1300V1',
            #data='Tax1300V1_2',
            #datas=['coco2017', 
                #'voc0712', 
                #'Materialist',
                #'Naturalist',
                #'Visual_Genome',
                #'brand1048',
                #'imagenet3k_448', 
                #'imagenet22k_448', 
                #'imagenet1kLoc',
                #'crawl_office_v1', 
                #'crawl_office_v2',
                #'mturk700_url_as_key'])
    #build_taxonomy_impl('./aux_data/taxonomy10k/tax700/Tax700V3',
            #data='Tax700V3_1',
            #datas=['coco2017', 
                #'voc0712', 
                #'brand1048',
                #'imagenet3k_448', 
                #'imagenet22k_448', 
                #'imagenet1kLoc',
                #'crawl_office_v1', 
                #'crawl_office_v2',
                #'mturk700_url_as_key'])
    #build_taxonomy_impl('./aux_data/taxonomy10k/Tax1300/Tax1300V1',
            #data='Tax1300SGV1_3',
            #datas=['coco2017',
                #'voc0712', 
                #'brand1048',
                #'imagenet3k_448',
                #'imagenet22k_448',
                #'imagenet1kLoc',
                #'mturk700_url_as_key',
                #'crawl_office_v1',
                #'crawl_office_v2',
                #'Materialist',
                #'VisualGenomeClean2',
                #'Naturalist'])
    #check_dataset()
    #remove_invalid_box()
    #build_taxonomy_impl('./aux_data/taxonomy10k/tax700/tax700_v2',
            #data='Tax700V3_1_debug',
            #datas=['Visual_Genome'])
    #philly_remove('data/qd_data/brand1048', vc='input')
    #philly_download('data/')
    #philly_ls('jianfw/work/qd_output', vc='input')
    #philly_ls('jianfw/work/qd_output/imagenet200_darknet19_448_B_noreorg_extraconv2/snapshot', vc='input')
    #philly_ls('sys/jobs/application_1514104910014_4627/models/ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_bb_only', vc='input')
    #philly_ls('sys/jobs/application_1514104910014_4627/models/ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_bb_nobb', vc='input')
    #philly_ls('sys/jobs/application_1514104910014_4627/models/ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_no_bb', vc='input')
    #philly_download('data/qd_data/Tax1300SGV1_1_no_bb', 'data/', vc='input')
    #philly_download('data/qd_data/Tax1300SGV1_1_with_bb', 'data/', vc='input')
    #philly_ls('code/', vc='input')
    #while True:
        #philly_download('work/qd_output', '/mnt/sdb/work/qd_output2', vc='input')
        #time.sleep(5 * 60)
    #from process_tsv import build_taxonomy_impl2
    #gen_coco_noffset_map()
    #test_visualize()
    #categoy_check()
    #build_taxonomy_impl2()
    #standarize_crawled('/raid/data/crawl_office_v1/TermList.pinterest.scrapping.image.tsv',
            #'/raid/data/crawl_office_v1/train.tsv')
    #build_taxonomy_impl('./aux_data/taxonomy10k/coco_phone_person',
            #datas=['coco2017'])
    #populate_dataset_details('Y')
    #extract_labels('coco2017', ['cell phone'], 'coco_phone')
    #design_massive()
    #create_imagenet3k_tsv()
    #create_imagenet22k_tsv()

    #d.get_trainval_tsv('inverted.label')
    #d = tsvdataset('imagenet3k_448')
    #d.get_trainval_tsv('inverted.label')
    #resize_dataset()
    #clean_tree()
    #populate_dataset_details('test')
    #analyze()
    pass

def num_cars():
    data = 'CARPK'
    split = 'test'
    data = 'CARPK_select.5.100000.nokeep'
    data = 'CARPK_select.5.50.nokeep'
    #data = 'CARPK'
    split = 'train'
    dataset = TSVDataset(data)
    label_file = dataset.get_data(split, 'label')
    rows = tsv_reader(label_file)
    all_num = []
    for row in rows:
        rects = json.loads(row[1])
        all_num.append(len(rects))
    x = np.asarray(all_num)
    logging.info('mean = {}'.format(np.mean(x)))
    logging.info('std = {}'.format(np.sqrt(np.var(x))))
    logging.info('total = {}'.format(np.sum(x)))

def check_mae_rmse():
    full_expid = 'CARPK_select.5.10.nokeep_darknet19_448_BCarWeight3_dataop8407_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    full_expid = 'CARPK_select.5.10.nokeep_MBTsvBoxSamples50.50.50.50.1CarWeight39_NMS0.2_darknet19_448_MBTsvBoxSamples50.50.50.50.1CarWeight3FullRun'
    full_expid = 'CARPK_select.5.50.nokeep_MBTsvBoxSamples50.50.50.50.1CarWeight39_NMS0.2_darknet19_448_MBTsvBoxSamples50.50.50.50.1CarWeight3FullRun'
    full_expid = 'CARPK_select.5.25.nokeep_MBTsvBoxSamples50.50.50.50.1CarWeight39_NMS0.2_darknet19_448_MBTsvBoxSamples50.50.50.50.1CarWeight3FullRun'
    full_expid = 'CARPK_select.5.5.nokeep_MBTsvBoxSamples50.50.50.50.1CarWeight39_NMS0.2_darknet19_448_MBTsvBoxSamples50.50.50.50.1CarWeight3FullRun'
    full_expid = 'CocoBottle1024DrinkY_select.10.5.nokeep_IBTsvBoxSamples50.19_NMS0.2_darknet19_448_IBTsvBoxSamples50.1FullRun'
    #full_expid = 'CocoBottle1024DrinkY_select.10.5.nokeep_darknet19_448_B_dataop8861_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1'
    full_expid = 'CocoBottle1024DrinkY_darknet19_448_B_dataop1453_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1'
    #full_expid = 'CARPK_select.5.5.nokeep_darknet19_448_B_dataop9450_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CARPK_select.5.25.nokeep_darknet19_448_BCarWeight3_dataop3107_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CARPK_select.5.50.nokeep_darknet19_448_BCarWeight3_dataop5427_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CARPK_darknet19_448_B_rotate0.0.10.180_r90.0.1.1.1_boxWeight.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CARPK_darknet19_448_B_dataop923_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CARPK_select.5.100000.nokeep_darknet19_448_B_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CARPK_select.5.100000.nokeep_darknet19_448_B_dataop4049_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CocoBottle1024DrinkY_darknet19_448_B_dataop1453_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1'
    #full_expid = 'CocoBottle1024DrinkY_select.10.5.nokeep_darknet19_448_B_dataop8861_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1'
    #full_expid = 'CARPK_select.5.100000.nokeep_darknet19_448_B_dataop4049_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    #full_expid = 'CocoBottle1024Drink_select.10.5.nokeep_IBTsvBoxSamples50.19_NMS0.2_darknet19_448_IBTsvBoxSamples50.1FullRun'

    get_best_mae(full_expid)


def get_best_mae(full_expid):
    net = 'darknet19_448'
    data_, expid_ = full_expid.split(net)
    data = data_[:-1]
    expid = expid_[1:]
    if data.startswith('CARPK'):
        test_data = 'CARPK'
    else:
        test_data = 'CocoBottle1024DrinkYW2'
    c = CaffeWrapper(data=data,
            net=net, 
            expid=expid,
            load_parameter=True,
            test_data=test_data)
    m = c.best_model()
    predict_file = c._predict_file(m)
    perf_file = c._perf_file(m)
    acc  = json.loads(read_to_buffer(perf_file + '.map.json'))
    logging.info('mAP = {}'.format(acc['overall']['0.5']))
    dataset = TSVDataset(test_data)
    fkey_to_gt, _ = load_labels(dataset.get_data('test', 'label'))
    fkey_to_pred, _ = load_labels(predict_file)
    
    len_testing_images = len(fkey_to_gt)
    absolute_error = np.zeros(len_testing_images)
    square_error = np.zeros(len_testing_images)
    step = 0.05
    all_th = np.asarray(range(1, int(1./step), 1)) * step
    all_mae = []
    all_rmse = []
    for th in all_th:
        for i, fkey in enumerate(fkey_to_gt):
            num_gt = len(fkey_to_gt[fkey])
            num_det = len(update_rects_within_image([l for l in
                fkey_to_pred[fkey] if l['conf'] > th]))
            absolute_error[i] = np.abs(num_det - num_gt)
            square_error[i] = (num_det - num_gt) ** 2
        MAE = np.mean(absolute_error)
        all_mae.append(MAE)
        #logging.info(th)
        #logging.info('Mean Absolute Error (MAE) = {}'.format(MAE))
        RMSE = np.sqrt(np.mean(square_error))
        all_rmse.append(RMSE)
        #logging.info('Root Mean Square Error (RMSE) = {}\n'.format(str(RMSE)))

    logging.info(min(zip(all_th, all_mae), key=lambda x: x[1]))
    logging.info(min(zip(all_th, all_rmse), key=lambda x: x[1]))
    
    logging.info('count = {}'.format(len_testing_images))

    return min(zip(all_th, all_mae), key=lambda x: x[1])

def put_in_image_bottom_left(curr_im, labels, color_mapper, data,
        font_thickness=None, font_scale=None):
    left = 5
    height = curr_im.shape[0] - 5
    if font_scale is None:
        font_scale = 2 if data.startswith('CAR') else 0.8
    if font_thickness is None:
        font_thickness = 2 if data.startswith('CAR') else 2
    labels = sorted(labels)
    for l in labels:
        text_width, text_height = put_text(curr_im, l, (left, height), 
                color_mapper[l],
                font_scale=font_scale,
                font_thickness=font_thickness)
        left = left + text_width + 5

def paper_figures():
    paper_figures_problem()
    #paper_figures_overfitting()
    #paper_figures_label_propogation()
    #paper_figures_visualize_result()
    #paper_figures_framework()
    pass

def paper_figures_framework():
    out_folder = '/home/jianfw/work/share/isl'
    data = 'CARPK_select.5.5.nokeep'
    pattern = 'CARPK_select.5.5.nokeep_MBNoBkgTsvBoxSamples50.50.50.50.1CarWeight3{}_NMS0.2'
    dataset = TSVDataset(data)
    key_to_idx = dataset.load_key_to_idx('train')
    all_key = key_to_idx.keys()
    all_data = []
    all_data.append(data)
    all_data.extend([pattern.format(i) for i in xrange(1, 10)])
    im = None
    for key in all_key:
        for i, data in enumerate(all_data):
            dataset = TSVDataset(data)
            key_to_idx = dataset.load_key_to_idx('train')
            idx = key_to_idx[key]
            tsv = TSVFile(dataset.get_data('train'))
            key2, str_rects, str_im = tsv.seek(idx)
            assert key == key2
            rects = json.loads(str_rects)
            im = img_from_base64(str_im)
            draw_bb(im, [r['rect'] for r in rects],
                    [r['class'] for r in rects],
                    color={'car': [0, 255, 255]},
                    rect_thickness=10,
                    draw_label=False)
            save_image(im, op.join(out_folder, 
                'framework_{}_{}.jpg'.format(key, i)))
        


def paper_figures_visualize_result():
    data = 'CARPK_select.5.5.nokeep'
    test_data = 'CARPK'
    #paper_figures_visualize_result_one(data, test_data)
    paper_figures_visualize_result_one(data, test_data, rect_thickness=5)

    #data = 'CocoBottle1024Merge_selectbylabel.10.1'
    #test_data = 'CocoBottle1024Merge'
    #paper_figures_visualize_result_one(data, test_data)

    data = 'CocoBottle1024Drink'
    test_data = 'CocoBottle1024DrinkYW2'
    paper_figures_visualize_result_one(data, test_data, rect_thickness=5)

def paper_figures_visualize_result_one(data, test_data, rect_thickness=2):

    net = 'darknet19_448'

    out_folder = '/home/jianfw/work/share/isl'
    all_prefix = ['base', 'full', 'ours']
    all_threshold = [0.1, 0.1, 0.1]
    # results from the baseline approach
    if data.startswith('CocoBottle1024Merge'):
        all_full_expid = []
        full_expid = 'CocoBottle1024Merge_selectbylabel.10.1_darknet19_448_B_dataop8126_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1'
        all_full_expid.append(full_expid)
        full_expid = 'CocoBottle1024Merge_selectbylabel.10.1000_darknet19_448_B_dataop2295_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1'
        all_full_expid.append(full_expid)
        full_expid = 'CocoBottle1024Merge_selectbylabel.10.1_IBTsvBoxSamples50.19_NMS0.2_darknet19_448_IBTsvBoxSamples50.1FullRun'
        all_full_expid.append(full_expid)
        keys = ['OldPicture_a_324',
                'OldPicture_f_291',
                'OldPicture_i_301',
                'OldPicture_d_37',
                'OldPicture_j_322']
    elif data.startswith('CocoBottle1024Drink'):
        all_full_expid = []
        all_full_expid.append('CocoBottle1024DrinkY_select.10.5.nokeep_darknet19_448_B_dataop8861_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1')
        all_full_expid.append('CocoBottle1024DrinkY_darknet19_448_B_dataop1453_randomScaleMin2_randomScaleMax4_TsvBoxSamples50.1')
        #all_full_expid.append('CocoBottle1024Drink_select.10.5.nokeep_IBTsvBoxSamples50.19_NMS0.2_darknet19_448_IBTsvBoxSamples50.1FullRun')
        all_full_expid.append('CocoBottle1024DrinkY_select.10.5.nokeep_IBTsvBoxSamples50.19_NMS0.2_darknet19_448_IBTsvBoxSamples50.1FullRun')
        keys = ['OldPicture_a_324',
                'OldPicture_f_291',
                'OldPicture_i_301',
                'OldPicture_e_290',
                'OldPicture_d_7',
                'OldPicture_d_37',
                'OldPicture_j_322']
        #all_threshold = [0.05, 0.5, 0.6]
    else:
        keys = ['20161225_TPZ_00158',
                '20161225_TPZ_00239',
                '20161225_TPZ_00156',
                '20161225_TPZ_00435']
        all_full_expid = []
        full_expid = 'CARPK_select.5.5.nokeep_darknet19_448_B_dataop9450_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
        all_full_expid.append(full_expid)
        full_expid = 'CARPK_select.5.100000.nokeep_darknet19_448_B_dataop4049_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
        all_full_expid.append(full_expid)
        full_expid = 'CARPK_select.5.5.nokeep_MBNoBkgTsvBoxSamples50.50.50.50.1CarWeight39_NMS0.2_darknet19_448_MBNoBkgTsvBoxSamples50.50.50.50.1CarWeight3FullRun'
        all_full_expid.append(full_expid)
        #all_threshold = [0.05, 0.5, 0.1]
    all_threshold = [float(get_best_mae(full_expid)[0]) for full_expid in all_full_expid]


    for full_expid, prefix, th in izip(all_full_expid, 
            all_prefix, all_threshold):
        data_, _expid = full_expid.split(net)
        data = data_[:-1]
        expid = _expid[1:]
        c = CaffeWrapper(data, net, load_parameter=True, 
                expid=expid,
                test_data=test_data)
        predict_file = c._predict_file(c.best_model())

        key_to_rects, _ = load_labels(predict_file)

        # upgrade the key to the full key
        all_full_key = key_to_rects.keys()
        for i in xrange(len(keys)):
            key = keys[i]
            found = False
            found_full_key = None
            for full_key in all_full_key:
                if key in full_key and not found:
                    found = True
                    found_full_key = full_key
                elif key in full_key and found:
                    raise Exception
            assert found
            keys[i] = found_full_key
        dataset = TSVDataset(test_data)
        test_label_file = dataset.get_data('test', 'label')
        _, key_to_idx = load_labels(test_label_file)
        test_file = dataset.get_data('test')
        tsv_test = TSVFile(test_file)
        predict_rects = [key_to_rects[key] for key in keys]
        label_mapper = {'FantaYellow': 'BottleFanta', 
                'ColaCan': 'BottleCola',
                'ColaPot': 'CanCola',
                'car': 'car',
                'drink': 'drink',
                'Sprite': 'CanSprite'}
        color_mapper = {'BottleFanta': [0, 255, 0],
                'CanSprite': [0, 0, 255],
                'car': [0, 255, 255],
                #'i': [0, 255, 255],
                'i': [0, 0, 255],
                'drink': [0, 255, 255],
                'BottleCola': [0, 255, 255],
                'CanCola': [255, 255, 0]}
        for key in keys:
            idx = key_to_idx[key]
            _, str_gt_rects, str_im = tsv_test.seek(idx)
            gt_rects = json.loads(str_gt_rects)
            im = img_from_base64(str_im)
            fname = 'vis_origin_{}_{}_{}'.format(test_data, prefix, key)
            fname = fname.replace('.', '_')
            save_image(im, op.join(out_folder, fname + '.jpg'))
            rects = [r for r in key_to_rects[key] if r['conf'] > th]
            update_rects_within_image(rects, im)
            for r in rects:
                r['class'] = label_mapper[r['class']]
            font_scale = 0.5 if data.startswith('CAR') else 0.5
            for r in rects:
                ious = [calculate_iou(r['rect'], g['rect']) for g in gt_rects]
                if max(ious) < 0.3:
                    r['class'] = 'i'
            rects = sorted(rects, key=lambda r: r['class'] == 'i')

            
            draw_bb(im, [r['rect'] for r in rects],
                    [r['class'] for r in rects],
                    #[r['conf'] for r in rects],
                    color=color_mapper,
                    font_scale=font_scale,
                    rect_thickness=rect_thickness,
                    font_thickness=1,
                    draw_label=False)
            font_scale = None
            if data.startswith('Coc'):
                if 'Merge' in data:
                    labels = ['BottleFanta', 'CanSprite', 'BottleCola', 'CanCola']
                else:
                    labels = ['drink']
                    font_scale = 2
            else:
                labels = ['car']
            put_in_image_bottom_left(im, labels, color_mapper, data,
                    font_thickness=5, font_scale=font_scale)
            meta = {}
            meta['th'] = th
            def count_by_label(rs):
                result = {}
                for r in rs:
                    if r['class'] in result:
                        result[r['class']] = result[r['class']] + 1
                    else:
                        result[r['class']] = 1
                return result
            meta['num_obj_gt'] = count_by_label(gt_rects)
            meta['num_predict'] = count_by_label(rects)
            meta['rects'] = rects
            fname = 'vis_{}_{}_{}'.format(test_data, prefix, key)
            fname = fname.replace('.', '_')
            save_image(im, op.join(out_folder, fname + '.jpg'))
            write_to_yaml_file(meta, op.join(
                out_folder + '_meta',
                fname + '.yaml'))

def paper_figures_label_propogation():
    #rect_thickness = 2
    #paper_figures_label_propogationOne(rect_thickness)
    rect_thickness = 10
    paper_figures_label_propogationOne(rect_thickness)

def paper_figures_label_propogationOne(rect_thickness):
    out_folder = '/home/jianfw/work/share/isl'
    data = 'CARPK_select.5.5.nokeep'
    pattern = 'CARPK_select.5.5.nokeep_MBNoBkgTsvBoxSamples50.50.50.50.1CarWeight3{}_NMS0.2'
    datas = [data]
    datas.extend([pattern.format(i) for i in range(1, 10)])
    select_ids = [0, 1, 2, 3, 4]
    dataset = TSVDataset(data)
    origin_dataset = TSVDataset(data)
    tsv = TSVFile(dataset.get_data('train'))
    select_rows = [tsv.seek(i) for i in select_ids]
    keys = dataset.load_keys('train')
    color_mapper = {'car': [0, 255, 255], 'car_i': [0, 0, 255]}
    key_to_rects, _ = load_labels(dataset.get_data('test', 'label'))
    key_to_selected_rects, _ = load_labels(dataset.get_data('train', 'label'))
    for j, data in enumerate(datas):
        dataset = TSVDataset(data)
        tsv = TSVFile(dataset.get_data('train'))
        for i in select_ids:
            key, str_rects, str_im = tsv.seek(i)
            rects = json.loads(str_rects)
            im = img_from_base64(str_im)
            update_rects_within_image(rects, im)
            put_in_image_bottom_left(im, set(r['class'] for r in rects),
                    color_mapper, 
                    data,
                    font_thickness=5)
            meta = {}
            meta['num_box'] = len(rects)
            gt_rects = key_to_rects[key]
            c = 0
            for r in rects:
                ious = [calculate_iou(r['rect'], g['rect']) for g in gt_rects]
                if max(ious) > 0.3:
                    c = c + 1
                else:
                    r['class'] = 'car_i'
            draw_bb(im, [r['rect'] for r in rects],
                    [r['class'] for r in rects],
                    color=color_mapper,
                    rect_thickness=rect_thickness,
                    draw_label=False)
            origin_rects = key_to_selected_rects[key]
            draw_bb(im, [r['rect'] for r in origin_rects],
                    [r['class'] for r in origin_rects],
                    color={'car': [255, 0, 0]},
                    rect_thickness=rect_thickness,
                    draw_label=False)
            save_image(im, op.join(out_folder, 
                'label_prop_{}_{}_{}.jpg'.format(key, rect_thickness, j)))
            meta['correct_box'] = c
            meta['total_box'] = len(gt_rects)
            write_to_yaml_file(meta, op.join(out_folder + '_meta', 
                'label_prop_{}_{}.yaml'.format(key, j)))


def paper_figures_overfitting():
    out_folder = '/home/jianfw/work/share/isl'
    data = 'CARPK_select.5.5.nokeep'
    full_expid = 'CARPK_select.5.5.nokeep_darknet19_448_B_dataop9450_rotate0.0.10.180.0_r90.0.1.1.1.0_boxWeight.1.1.1.1.1_randomScaleMin2_randomScaleMax4'
    net = 'darknet19_448'
    data, expid = full_expid.split('_{}_'.format(net))
    c = CaffeWrapper(data, net, load_parameter=True,
            expid=expid,
            test_data=data)
    select_ids = [0, 1]
    dataset = TSVDataset(data)
    keys = dataset.load_keys('train')
    select_keys = [keys[i] for i in select_ids]
    tsv = TSVFile(dataset.get_data('train'))
    select_rows = [tsv.seek(i) for i in select_ids]
    predict_file = c._predict_file(c.best_model())
    key_to_rects, key_to_idx = load_labels(predict_file)
    def put_in_image_bottom_left(curr_im, labels, color_mapper):
        left = 5
        height = curr_im.shape[0] - 5
        font_scale = 2 if data.startswith('CAR') else 0.8
        font_thickness = 5 if data.startswith('CAR') else 2
        labels = sorted(labels)
        for l in labels:
            text_width, text_height = put_text(curr_im, l, (left, height), 
                    color_mapper[l],
                    font_scale=font_scale,
                    font_thickness=font_thickness)
            left = left + text_width + 5
    color_mapper = {'car': [0, 255, 255]}
    for row in select_rows:
        key = row[0]
        pred_rects = key_to_rects[key]
        im = img_from_base64(row[-1])
        #gt_labels = json.loads(row[1])
        #draw_bb(im, [r['rect'] for r in gt_labels], [r['class'] for r in
            #gt_labels], color={'car': [0, 0, 255]})
        filtered_pred_rects = [r for r in pred_rects if r['conf'] > 0.05]
        draw_bb(im, [r['rect'] for r in filtered_pred_rects],
                [r['class'] for r in filtered_pred_rects], 
                [r['conf'] for r in filtered_pred_rects], 
                rect_thickness=7,
                font_thickness=3,
                font_scale=1.5,
                draw_label=False,
                color=color_mapper)
        put_in_image_bottom_left(im, set(r['class'] for r in
            filtered_pred_rects), color_mapper=color_mapper)
        save_image(im, op.join(out_folder, 'overfitting_' + key + '.jpg'))


def paper_figures_problem():
    data = 'CARPK_select.5.5.nokeep'
    full_data = 'CARPK'
    color_mapper = {'car': [0, 255, 255]}
    for idx in xrange(5):
        paper_figures_different_training(data, full_data, idx,
                color_mapper=color_mapper)

    data = 'CocoBottle1024Merge_selectbylabel.10.1'
    full_data = 'CocoBottle1024Merge'
    label_mapper = {'FantaYellow': 'BottleFanta', 
            'ColaCan': 'BottleCola',
            'ColaPot': 'CanCola',
            'Sprite': 'CanSprite'}
    color_mapper = {'BottleFanta': [0, 255, 0],
            'CanSprite': [0, 0, 255],
            'BottleCola': [0, 255, 255],
            'CanCola': [255, 255, 0]}
    paper_figures_different_training(data, full_data, 9, label_mapper,
            keep=0.575, color_mapper=color_mapper)

    data = 'CocoBottle1024DrinkY_select.10.5.nokeep'
    full_data = 'CocoBottle1024DrinkY'
    color_mapper = {'drink': [0, 255, 255]}
    paper_figures_different_training(data, full_data, 9, 
            label_mapper=None,
            keep=1, 
            color_mapper=color_mapper)

def paper_figures_different_training(data, full_data, idx, 
        label_mapper=None, keep=1, color_mapper=None):
    #paper_figures_different_trainingv1(data, full_data,
            #idx, label_mapper, keep, color_mapper)
    paper_figures_different_trainingv2(data, full_data,
            idx, label_mapper, keep, color_mapper)

def paper_figures_different_trainingv1(data, full_data, idx, 
        label_mapper=None, keep=1, color_mapper=None):
    dataset = TSVDataset(data)
    from process_image import save_image
    from process_image import put_text
    out_folder = '/home/jianfw/work/share/isl'
    train_file = dataset.get_data('train')
    tsv = TSVFile(train_file)
    key, str_rect, str_im = tsv.seek(idx)
    rects = json.loads(str_rect)
    im = img_from_base64(str_im)
    if keep < 1:
        keep_height = int(im.shape[0] * keep)
        im = im[:keep_height, :, :]
    if label_mapper:
        for r in rects:
            r['class'] = label_mapper[r['class']]
    
    # load the image with full label
    full_dataset = TSVDataset(full_data)
    full_key_to_idx = full_dataset.load_key_to_idx('train')
    full_label_file = full_dataset.get_data('train', 'label')
    _, str_full_rects, _ = TSVFile(full_dataset.get_data('train')).seek(full_key_to_idx[key])
    full_rects = json.loads(str_full_rects)
    if label_mapper:
        for r in full_rects:
            r['class'] = label_mapper[r['class']]
    curr_im = np.copy(im)
    draw_bb(curr_im, [r['rect'] for r in full_rects],
            [r['class'] for r in full_rects],
            font_scale=0.7 if data.startswith('CARPK') else 0.3,
            font_thickness=1,
            color=color_mapper)
    save_image(curr_im, op.join(out_folder, 
        '{}_{}fsl.jpg'.format(full_data, '' if keep >= 1 else '{}_'.format(keep))))

    curr_im = np.copy(im)
    logging.info(', '.join(set(r['class'] for r in rects)))
    put_text(curr_im, 
            ', '.join(set(r['class'] for r in rects)),
            (5, curr_im.shape[0] - 5), 
            (0, 255, 255),
            font_scale=2 if data.startswith('CAR') else 1,
            font_thickness=2)
    save_image(curr_im, op.join(out_folder, 
        '{}_{}wsl.jpg'.format(full_data, '' if keep >= 1 else '{}_'.format(keep))))

    curr_im = np.copy(im)
    draw_bb(curr_im, [r['rect'] for r in rects],
            [r['class'] for r in rects],
            font_scale=1,
            font_thickness=2, 
            color=color_mapper)
    save_image(curr_im, op.join(out_folder, 
        '{}_{}isl.jpg'.format(full_data, '' if keep >= 1 else '{}_'.format(keep))))

def paper_figures_different_trainingv2(data, full_data, idx, 
        label_mapper=None, keep=1, color_mapper=None):
    dataset = TSVDataset(data)
    out_folder = '/home/jianfw/work/share/isl'
    train_file = dataset.get_data('train')
    tsv = TSVFile(train_file)
    key, str_rect, str_im = tsv.seek(idx)
    rects = json.loads(str_rect)
    im = img_from_base64(str_im)
    if keep < 1:
        keep_height = int(im.shape[0] * keep)
        im = im[:keep_height, :, :]
    if label_mapper:
        for r in rects:
            r['class'] = label_mapper[r['class']]

    def put_in_image_bottom_left(im, labels, color_mapper):
        left = 5
        height = im.shape[0] - 5
        font_scale = 3 if data.startswith('CAR') else 2
        font_thickness = 5 if data.startswith('CAR') else 5
        labels = sorted(labels)
        for l in labels:
            text_width, text_height = put_text(curr_im, l, (left, height), 
                    color_mapper[l],
                    font_scale=font_scale,
                    font_thickness=font_thickness)
            left = left + text_width + 5
    
    # load the image with full label
    full_dataset = TSVDataset(full_data)
    full_key_to_idx = full_dataset.load_key_to_idx('train')
    full_label_file = full_dataset.get_data('train', 'label')
    _, str_full_rects, _ = TSVFile(full_dataset.get_data('train')).seek(full_key_to_idx[key])
    full_rects = json.loads(str_full_rects)
    if label_mapper:
        for r in full_rects:
            r['class'] = label_mapper[r['class']]
    if data.startswith('Coco'):
        full_rects.append({'class': 'CanCola', 'rect': [172, 115, 209, 162]})
        full_rects.append({'class': 'CanCola', 'rect': [213, 117, 252, 159]})
        full_rects.append({'class': 'CanCola', 'rect': [255, 116, 292, 159]})
        full_rects.append({'class': 'CanCola', 'rect': [293, 115, 333, 159]})
        full_rects.append({'class': 'CanSprite', 'rect': [337, 117, 373, 161]})
        full_rects.append({'class': 'CanSprite', 'rect': [375, 113, 418, 161]})
    curr_im = np.copy(im)
    rect_thickness = 5 if data.startswith('Co') else 5
    full_rects = [r for r in full_rects if r['class'] == 'car' or r['class'] ==
        'drink']
    draw_bb(curr_im, [r['rect'] for r in full_rects],
            [r['class'] for r in full_rects],
            font_scale=0.7 if data.startswith('CARPK') else 0.3,
            font_thickness=1,
            rect_thickness=rect_thickness,
            color=color_mapper,
            draw_label=False)
    put_in_image_bottom_left(curr_im, set(r['class'] for r in rects), 
            color_mapper)
    save_image(curr_im, op.join(out_folder, 
        '{}_{}fsl.jpg'.format(full_data, 
            '' if keep >= 1 else '{}_'.format(keep).replace('.', '_'),
            idx)))

    curr_im = np.copy(im)
    put_in_image_bottom_left(curr_im, set(r['class'] for r in rects), 
            color_mapper)
    save_image(curr_im, op.join(out_folder, 
        '{}_{}wsl.jpg'.format(full_data, 
            '' if keep >= 1 else '{}_'.format(keep).replace('.', '_'))))

    curr_im = np.copy(im)
    draw_bb(curr_im, [r['rect'] for r in rects],
            [r['class'] for r in rects],
            font_scale=1,
            font_thickness=2, 
            color=color_mapper,
            rect_thickness=10,
            draw_label=False)
    put_in_image_bottom_left(curr_im, set(r['class'] for r in rects), 
            color_mapper)
    save_image(curr_im, op.join(out_folder, 
        '{}_{}isl_{}.jpg'.format(full_data, 
            '' if keep >= 1 else '{}_'.format(keep).replace('.', '_'),
            idx)))

def extract_image_for_label():
    from process_tsv import ImageTypeParser
    def gen_rows():
        data = 'CocoBottle1024Drink_select.10.10000.nokeep'
        dataset = TSVDataset(data)
        fname = dataset.get_data('train')
        rows = tsv_reader(fname)
        #for row in rows:
            #yield row
        data = 'CocoBottle1024Drink'
        dataset = TSVDataset(data)
        rows = tsv_reader(dataset.get_data('test'))
        for row in rows:
            yield row
    p = ImageTypeParser()
    for key, str_rects, str_encoded_im in gen_rows():
        str_im = base64.b64decode(str_encoded_im)
        t = p.parse_type(str_im)
        write_to_file(str_im,
                op.join('/home/jianfw/code/BBox-Label-Tool/Images/003',
                    '{}.JPEG'.format(key)))
        rects = json.loads(str_rects)
        all_line = []
        all_line.append(str(len(rects)))
        for r in rects:
            all_line.append(' '.join(map(str, r['rect'])))
        write_to_file('\n'.join(all_line),
                op.join('/home/jianfw/code/BBox-Label-Tool/Labels/003',
                    '{}.txt'.format(key)))


def create_coca1024drink_yan():
    label_tool_root = '/home/jianfw/code/BBox-Label-Tool'
    src_train_data = 'CocoBottle1024Drink_select.10.10000.nokeep'
    src_test_data = 'CocoBottle1024Drink'
    dest_data = 'CocoBottle1024DrinkYW2'
    dataset = TSVDataset(src_train_data)
    fname = dataset.get_data('train')
    def gen_rows(fname):
        rows = tsv_reader(fname)
        for row in rows:
            key = row[0]
            str_encoded_im = row[-1]
            jpg_name = op.join(label_tool_root, 
                    'Images', 
                    '003', 
                    '{}.JPEG'.format(key))
            assert op.isfile(jpg_name)
            label_fname = op.join(label_tool_root, 'Labels', '003', 
                    '{}.txt'.format(key))
            assert op.isfile(label_fname)
            all_line = load_list_file(label_fname)
            assert len(all_line) > 0
            num = int(all_line[0])
            assert len(all_line) == num + 1
            rects = []
            for i in xrange(1, num + 1):
                line = all_line[i]
                rect = [float(s) for s in line.split()]
                rects.append({'class': 'drink', 'rect': rect})
            row[1] = json.dumps(rects)
            yield row
    dest_dataset = TSVDataset(dest_data)
    tsv_writer(gen_rows(fname), dest_dataset.get_data('train'))
    dataset = TSVDataset(src_test_data)
    fname = dataset.get_data('test')
    tsv_writer(gen_rows(fname), dest_dataset.get_data('test'))
    populate_dataset_details(dest_data)

def towards_incomplete():
    from qd_util import resize_tsv
    from qd_util import merge_labels
    from qd_util import create_cocobottle
    from qd_util import create_logs18
    from qd_util import check_wider_face
    from qd_util import upload_carpk_to_philly
    from qd_util import create_wider_face
    from qd_util import create_toy_dataset
    #test_add_prediction_into_train()
    #create_toy_dataset()
    #yolo_incomplete_label()
    #populate_dataset_details('CARPK_select.5.100000.nokeep')
    #incomplete_iterative()
    #create_coca1024drink_yan()
    #paper_figures()
    #extract_image_for_label()
    #check_mae_rmse()
    #num_cars()
    #create_wider_face()
    #test_dataset_op_select()
    #upload_carpk_to_philly()
    #check_wider_face()
    #from qd_util import check_target_overlap
    #build_taxonomy_impl('./aux_data/tile',
            #data='Tile',
            #datas=['imagenet3k_448'],
            #max_image_per_label=100000,
            #min_image_per_label=1)
    #populate_dataset_details('Tile_with_bb')
    #create_logs18()
    #create_cocobottle()
    #merge_labels()
    #resize_tsv()
    #check_target_overlap()
    #resize_tsv()
    #pipe_run()
    #count_car()
    add_prediction()

def add_prediction():
    data = 'voc20'
    expid = 'B'
    net = 'darknet19_448'
    c = CaffeWrapper(data=data,
            test_data=data,
            net=net,
            expid=expid,
            load_parameter=True)
    m = c.best_model()
    yolo_nms = 0.3
    out_data = 'voc20ISL'
    predict_file = c._predict_file(m)
    data = add_prediction_into_train(data, 
            predict_file, 
            yolo_nms,
            out_data)

def submit_job():
    import requests
    jobparams = """
        {
        "jobName":"caffe-ssh",
        "resourcegpu":0,
        "workPath":"./",
        "dataPath":"imagenet",
        "jobPath":"",
        "image":"bvlc/caffe:cpu",
        "cmd":"echo $HOME",
        "interactivePort":"22",
        "runningasroot":true,
        "env":[],
        "jobtrainingtype":"RegularJob"
        }
        """
    payload = {}
    payload["Json"] = jobparams
    email = '"jianfw@microsoft.com"'
    #email = 'jianfw'
    key = 'HappyBirthday88'
    #key = '0e7e2eca'
    url = 'http://hongzltest-infra01.westus2.cloudapp.azure.com/api/dlws/postJob?Email={}&Key={}'.format(
            email,
            key)
    logging.info(url)
    r = requests.post(url, data=payload)
    logging.info(r.text)

def check_two_prediction():
    official_predict_file = \
        'output/imagenet2012_SEBNInception_Official/snapshot/model_iter_450000.caffemodel.imagenet2012.test.predictAstsvdatalayer.predict'
    our_predict_file = \
            'output/imagenet2012_SEBNInception_A/snapshot/model_iter_450000.caffemodel.imagenet2012.test.predictAstsvdatalayer.predict'

    def load_predict(predict_file):
        rows = tsv_reader(predict_file)
        name_to_scores = {}
        for row in rows:
            assert row[0] not in name_to_scores
            name_to_scores[row[0]] = map(float, row[1].split(','))
        return name_to_scores
    logging.info('loading {}'.format(official_predict_file))
    official_predict = load_predict(official_predict_file)
    our_predict = load_predict(our_predict_file)
    for name in official_predict:
        curr_official = official_predict[name]
        curr_our = our_predict[name]
        curr_off_argmax = np.argmax(curr_official)
        curr_our_argmax = np.argmax(curr_our)
        import ipdb;ipdb.set_trace(context=15)


def test():
    from remote_run import scp_f
    from qd_util import netprototxt_to_netspec
    from qd_util import logic_task
    #x = load_binary_net('output/imagenet2012_SEBNInception_Official/snapshot/model_iter_450000.caffemodel')
    #import ipdb;ipdb.set_trace(context=15)
    #rows = tsv_reader('output/imagenet2012_SEBNInception_Official/snapshot/model_iter_450000.caffemodel.imagenet2012.test.predictAstsvdatalayer.predict.tmp.tmp')
    #for row in rows:
        #import ipdb;ipdb.set_trace(context=15)
    #real_data = read_blob('/home/jianfw/real_data.bin')
    #pre_data = read_blob('/home/jianfw/preprocessing_data.bin')
    #all_image = network_input_to_image(real_data, [104, 117, 123])
    #show_image(all_image[0])
    #import ipdb;ipdb.set_trace(context=15)
    #submit_job()
    #test_correct_caffe_file_path()
    #test_parse_log()
    #test_caffemodel_num_param()
    #test_parallel_train()
    #test_calculate_macc()
    # -------------- misc
    #get_all_data_info()
    #parameters = get_parameters_by_full_expid('CARPK_darknet19_448_A_fullGpu_randomScaleMin2_randomScaleMax4_maxIter.600e')
    #logging.info(pformat(parameters))
    #populate_dataset_details('t700_mturk_test')
    #replace_empty_by_zerolist('imagenet200', 'imagenet200_2')
    #extract_dataset('imagenet2012')
    #populate_dataset_details('imagenet2012')

    #yolo_old_to_new_test()
    #pr_curve()
    #test_ssd() logic_task()
    #logic()
    #test_dataset_op_removelabel()

    #all_val = load_list_file('/home/jianfw/code/quickdetection/src/CCSCaffe/data/ilsvrc12/val.txt')
    #all_val = [val.split(' ') for val in all_val]
    #all_val = [(name, int(label)) for name, label in all_val]
    #rows = tsv_reader('data/imagenet2012/test.label.tsv')
    #name_to_label = {}
    #for row in rows:
        #assert row[0] not in name_to_label
        #name_to_label[row[0]] = int(row[1])
    #import ipdb;ipdb.set_trace(context=15)
    #for name, caffe_label in all_val:
        #assert caffe_label == name_to_label[name]
    
    #check_two_prediction()
    
    # --------------------- copy data
    #ssh_info = {
        #'username': 'jianfw',
        #'ip': '10.228.131.87',
        #'-p': 2289,
        #'-i': '/mnt/rr1.pnrsy/sys/jobs/application_1523653358074_9667/.ssh/id_rsa'}
    #ssh_info = {
        #'username': 'jianfw',
        #'ip': '10.228.131.87',
        #'-p': 2289,
        #'-i': '/mnt/rr1.pnrsy/application_1523653358074_9667/id_rsa'}
    #upload_qdoutput('output/TaxVocPerson_V1_1_darknet19_person_bb_only',
            #'/hdfs/pnrsy/jianfw/work/qd_output/TaxVocPerson_V1_1_darknet19_person_bb_only',
            #ssh_info)
    #scp_f('/mnt/jianfw_desk/libcudnn7_7.1.3.16-1+cuda8.0_amd64.deb', 
            #'/tmp/code/',
            #ssh_info)
    #scp_f('/mnt/jianfw_desk/libcudnn7-dev_7.1.3.16-1+cuda8.0_amd64.deb', 
            #'/tmp/code/',
            #ssh_info)
    #scp_f('data/clothingClean', '/hdfs/pnrsy/jianfw/data/qd_data/',
            #ssh_info)
    #populate_dataset_details('imagenet2012')

    #netprototxt_to_netspec()

    # ------------------ incomplete label project
    #generate_carpk_dataset()
    #generate_pipe_dataset()

    # ------------------- analize the performance
    #force_negative_visualization()

    # -------------------- train imagenet200
    #yolo_imagenet200()
    
    # ----------------- philly
    #philly_upload_dir('data/voc20', 'data/qd_data')
    #philly_ls('data/qd_data/')
    #philly_ls('code/')
    #philly_download('code/test.bash', './')
    #philly_mkdir('data/qd_data')
    #philly_upload('/home/jianfw/code/tmp/a.py', 'data')
    pass

def test_parse_log():
    file_name = \
    '/home/jianfw/code/quickdetection/output/voc20_darknet19_baseline/darknet19_20170710_205121.log20170710-205127.8203'
    x, y = parse_training_time(file_name)

def test_caffemodel_num_param():
    caffemodel_num_param(m)

def test_parallel_train():
    solver = './output/voc20_darknet19_baseline_2/solver.prototxt'
    solver = './output/tmp/solver.prototxt'
    snapshot = None
    weights = './models/darknet19.caffemodel'
    gpus = [0,1,2,3]
    parallel_train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        weights,
        gpus,  # list of device ids
        timing=False)

def test_calculate_macc():
    proto = \
    './output/imagenet2012_resnet10_B_sl0.1si0.1_weightdecay0.0001/test.prototxt'
    #logging.info(calculate_macc(proto)/1000000.)
    logging.info(process_run(calculate_macc, proto) / 1000000.)


def towards_msoftmax():
    #through_mxnet()
    populate_dataset_details('cifar100')

def through_mxnet():
    num_real_classes = 10
    num_classes = 2 # the first 2 classes
    num_train_sample = 1000
    num_test_sample = 100
    radius = 10
    sigma = 1
    batch = 64
    
    degree_per_class = 360. / num_real_classes
    sigma_maxtrix = np.zeros((2, 2))
    sigma_maxtrix[0, 0] = sigma * sigma
    sigma_maxtrix[-1, -1] = sigma * sigma
    num_train_sample_per_real_class = num_train_sample / num_real_classes
    num_test_sample_per_real_class = num_test_sample / num_real_classes
    
    def create_feat_label(num_sample_per_real_class):
        features = []
        labels = []
        for c in range(num_real_classes):
            curr_degree = c * degree_per_class
            mu_x = radius * np.cos(curr_degree / 180. * np.pi)
            mu_y = radius * np.sin(curr_degree / 180. * np.pi)
        
            curr_samples = np.random.multivariate_normal([mu_x, mu_y],
                    sigma_maxtrix, num_sample_per_real_class)
            curr_labels = np.zeros((num_sample_per_real_class))
            curr_labels[:] = c
            features.append(curr_samples)
            labels.append(curr_labels)
        feat, label = np.vstack(features), np.hstack(labels)
        n = len(feat)
        idx = range(n)
        import random
        random.shuffle(idx)
        return feat[idx, :], label[idx]
    train_features, train_labels = create_feat_label(num_train_sample_per_real_class)
    test_features, test_labels = create_feat_label(num_test_sample_per_real_class)

    def mask_labels(labels):
        '''
        make all other classes except the first num_classes as num_classes,
        which means the background
        '''
        labels[labels >= num_classes] = num_classes


    # construct a one-layer network
    def train():
        import mxnet as mx
        num_inferred_class = len(set(train_labels))
        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data, name='fc1', 
                num_hidden=num_inferred_class,
                no_bias=True)
        out = mx.sym.SoftmaxOutput(fc1, name='softmax')
        mod = mx.mod.Module(out)
        train_iter = mx.io.NDArrayIter(data=train_features, 
                label=train_labels,
                batch_size=batch)
        mod.bind(data_shapes=train_iter.provide_data, 
                label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
        mod.init_optimizer(optimizer='sgd', 
                optimizer_params=(('learning_rate', 0.1), ))
        mod.fit(train_iter, num_epoch=50)
        params, aux_params = mod.get_params()
        
        colors = []
        for c in range(num_inferred_class):
            from random import random
            colors.append([random(), random(), random()])
        # training samples
        for c in range(num_inferred_class):
            is_curr = train_labels == c
            plt.scatter(train_features[is_curr, 0], 
                    train_features[is_curr, 1],
                    c=colors[c])

        w = params['fc1_weight'].asnumpy()
        for i, (x, y) in enumerate(w):
            ratio = radius / np.sqrt(x * x + y * y)
            plt.plot([0, ratio * x], [0, ratio * y], color=colors[i])
            plt.scatter([x], [y], color=colors[i])
        plt.grid()
        plt.show()

    #train()
    mask_labels(train_labels)
    #train()

    import mxnet as mx
    data = mx.sym.Variable('data')
    fc1 = mx.sym.FullyConnected(data, name='fc1', 
            num_hidden=num_real_classes,
            no_bias=True)
    softmax_full = mx.sym.Softmax(fc1, name='softmax')
    first_positive = mx.sym.slice(softmax_full, begin=0, end=2)
    negatives = mx.sym.slice(softmax_full, begin=2)
    sum_neg = mx.sym.sum(negatives)
    pos_neg = mx.sym.concat(first_positive, sum_neg)
    label = mx.sym.Variable('label')
    out = mx.sym.make_loss(mx.sym.slice(mx.sym.log(pos_neg), begin=label))
    mod = mx.mod.Module(out)
    train_iter = mx.io.NDArrayIter(data=train_features, 
            label=train_labels,
            batch_size=batch)
    mod.bind(data_shapes=train_iter.provide_data, 
            label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(optimizer='sgd', 
            optimizer_params=(('learning_rate', 0.1), ))
    mod.fit(train_iter, num_epoch=50)
    params, aux_params = mod.get_params()
    
    colors = []
    for c in range(num_inferred_class):
        from random import random
        colors.append([random(), random(), random()])
    # training samples
    for c in range(num_inferred_class):
        is_curr = train_labels == c
        plt.scatter(train_features[is_curr, 0], 
                train_features[is_curr, 1],
                c=colors[c])

    w = params['fc1_weight'].asnumpy()
    for i, (x, y) in enumerate(w):
        ratio = radius / np.sqrt(x * x + y * y)
        plt.plot([0, ratio * x], [0, ratio * y], color=colors[i])
        plt.scatter([x], [y], color=colors[i])
    plt.grid()
    plt.show()


def towards_tracking():
    from qd_util import count_num_alov300
    from qd_util import init_last_conv_by_min_l2
    from qd_util import create_vot_dataset
    from qd_util import test_init_last_conv_by_min_l2
    #create_vot_dataset()
    #count_num_alov300()
    test_init_last_conv_by_min_l2()

def philly():
    #philly_remove('data/qd_data/brand1048', vc='input')
    #philly_ls('jianfw/work/qd_output', vc='input')
    #philly_ls('jianfw/work/qd_output/imagenet200_darknet19_448_B_noreorg_extraconv2/snapshot', vc='input')
    #philly_ls('sys/jobs/application_1514104910014_4627/models/ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_bb_only', vc='input')
    #philly_ls('sys/jobs/application_1514104910014_4627/models/ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_bb_nobb', vc='input')
    #philly_ls('sys/jobs/application_1514104910014_4627/models/ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_no_bb', vc='input')
    #philly_download('data/qd_data/Tax1300SGV1_1_no_bb', 'data/', vc='input')
    from qd_util import philly_download_qdoutput
    from qd_util import parse_philly_ls_output
    from qd_util import update_config
    from qd_util import philly_upload_qdoutput
    from qd_util import philly_remove
    #update_config()
    #parse_philly_ls_output(s)
    #philly_remove('jianfw/work/qd_output/Tax700V3_1_darknet19_448_B_noreorg_extraconv2_tree_init5494_IndexLossWeight0_bb_nobb')
    #philly_remove('code')
    #philly_ls('.')
    #full_expid = 'Tax1300SGV1_1_darknet19_448_B_noreorg_extraconv2_tree_init3491_IndexLossWeight0_bb_only'
    #philly_download_qdoutput(op.join('jianfw', 'work', 'qd_output',
        #full_expid),
        #op.join('output', full_expid))

    #full_expid = 'Tax4k_V1_2_darknet19_no_visual_genome_bb_nobb'
    #philly_upload_qdoutput('output/{}'.format(full_expid),
            #'jianfw/work/qd_output/{}'.format(full_expid))
    #philly_ls('jianfw/data/qd_data/voc20')
    #philly_download_qdoutput('jianfw/work/qd_output/imagenet200_darknet19_448_B_noreorg_extraconv2',
            #'output/imagenet200_darknet19_448_B_noreorg_extraconv2')
    #full_expid = 'voc20_darknet19_448_C_noreorg_extraconv2_tree'
    #philly_download_qdoutput('jianfw/work/qd_output/{}'.format(full_expid),
            #'output/{}'.format(full_expid))
    #philly_download('work/qd_output/imagenet200_darknet19_448_B_noreorg_extraconv2',
            #'output/imagenet200_darknet19_448_B_noreorg_extraconv2',
            #vc='input',
            #user='jianfw')
    #philly_download('jianfw/data/qd_data/Tax1300SGV1_2_with_bb', 'data/', vc='input')
    #philly_download('jianfw/data/qd_data/Tax1300SGV1_2_no_bb', 'data/', vc='input')
    #philly_download('jianfw/data/qd_data/Tax1300SGV1_2', 'data/', vc='input')
    
    #philly_ls('jianfw/data/qd_data')
    #while True:
        #philly_ls('jianfw/data/qd_data/TaxPerson_V1_2_with_bb_S_M1')
        #time.sleep(5)
    philly_upload_dir('data/TaxPerson_V1_2_with_bb', 
            'jianfw/data/qd_data/',
            vc='input')
    #philly_upload_dir('data/voc20', 
            #'jianfw/data/qd_data',
            #vc='input',
            #cluster='eu2')
    # examples
    #philly_upload_dir('data/voc20', 
            #'jianfw/data/qd_data/',
            #vc='input')
    #philly_upload_dir('output/Tax700V2_darknet19_448_B_noreorg_extraconv2_bb_only', 
            #'work/qd_output/',
            #vc='input')
    #philly_ls('jianfw/data/qd_data/', vc='input')
    #while True:
        #philly_download('work/qd_output', '/mnt/sdb/work/qd_output2', vc='input')
        #time.sleep(5 * 60)
    #philly_upload_dir('data/imagenet1kLoc', 
            #'data/qd_data',
            #vc='input')
    #philly_upload_dir('data/Visual_Genome', 
            #'jianfw/data/qd_data/',
            #vc='input')
    #philly_upload_dir('data/Naturalist', 
            #'jianfw/data/qd_data/',
            #vc='input')
    #philly_upload_dir('output/Tax700V2_darknet19_448_B_noreorg_extraconv2_bb_only', 
            #'work/qd_output/',
            #vc='input')
    pass

if __name__ == '__main__':
    init_logging()
    random.seed(777)
    #remove_bb_train_test()
    #remove_bb_train_test2()
    #yolo9000_coco50K()
    #yolo9000()
    #study_loss_per_cat()
    #smaller_network_input_size()

    #check_yolo_full_gpu()
    #check_yolo_test_full_gpu()

    #low_shot_checking()
    #compare()

    #compare_log_for_multibin()
    #study_target()
    #test_()
    #visualize_multibin()
    #visualize_multibin2()

    #test_demo()
    #all_flops()
    #mobile_net()
    #test_yolo9000_on_imagenet()
    #cifar()
    #print_label_order()
    #test()
    #towards_msoftmax()
    #towards_tracking()
    torwards_10K()
    #philly()
    #towards_incomplete()

    #classification_task()
    #yolo_master()
    #yolo_demo()

    #re_run()
    #test_devonc()
    #project_pipe()
    #generate_pipe_dataset()

