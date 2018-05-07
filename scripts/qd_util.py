import yaml
import numpy as np
import shutil
from numpy import linalg as la
import os.path as op
import glob
import os
from process_tsv import populate_dataset_details
from tsv_io import TSVDataset, TSVFile, tsv_writer, tsv_reader
import json
import random
from qd_common import img_from_base64
from process_tsv import show_image, draw_bb
import cv2
from qd_common import encoded_from_img
from remote_run import cmd_run
import urllib2
import logging
import copy
import struct
from taxonomy import Taxonomy
from qd_common import load_list_file
from process_image import show_image
from itertools import izip
import base64
from qd_common import read_to_buffer
from email_util import notify_by_email
from gpu_util import gpu_available
from remote_run import sync_qd
from qd_common import load_solver
from qd_common import ensure_directory
import re
from multiprocessing import Queue
from pprint import pformat
import time
from qd_common import load_from_yaml_file
from itertools import izip
from process_image import show_images
import shutil
import glob
import yaml
import magic
import matplotlib.pyplot as plt
from tsv_io import tsv_reader, tsv_writer
from pprint import pformat
import os
import os.path as op
import sys
import json
from pprint import pprint
import multiprocessing as mp
import random
import numpy as np
from qd_common import ensure_directory
from qd_common import default_data_path, load_net
import time
from multiprocessing import Queue
from shutil import copytree
from shutil import rmtree
from qd_common import img_from_base64
import cv2
import base64
from taxonomy import load_label_parent, labels2noffsets
from taxonomy import LabelToSynset, synset_to_noffset
from taxonomy import noffset_to_synset
from taxonomy import get_nick_name
from taxonomy import load_all_tax, merge_all_tax
from taxonomy import child_parent_print_tree2
from taxonomy import create_markdown_url
#from taxonomy import populate_noffset
from taxonomy import populate_cum_images
from taxonomy import gen_term_list
from taxonomy import gen_noffset
from taxonomy import populate_url_for_offset
from taxonomy import disambibuity_noffsets
from taxonomy import Taxonomy
from qd_common import read_to_buffer, load_list_file
from qd_common import write_to_yaml_file, load_from_yaml_file
from qd_common import encoded_from_img
from tsv_io import extract_label
from tsv_io import create_inverted_tsv
from qd_common import process_run

from process_image import draw_bb, show_image, save_image
from qd_common import write_to_file
from tsv_io import TSVDataset, TSVFile
from qd_common import init_logging
import logging
from qd_common import is_cluster
from tsv_io import tsv_shuffle_reader
import argparse
from tsv_io import get_meta_file
import imghdr
from qd_common import calculate_iou
from qd_common import yolo_old_to_new
from qd_common import generate_lineidx
from qd_common import list_to_dict
from qd_common import dict_to_list
from qd_common import parse_test_data
from tsv_io import load_labels
import caffe
from qd_common import network_input_to_image
from qd_common import calculate_ap_by_true_list 
from taxonomy import is_noffset
from remote_run import remote_run
from remote_run import scp

def list_bool_vector(num_atom):
    assert num_atom > 0
    if num_atom == 1:
        return [[False], [True]]
    else:
        result = []
        sub = list_bool_vector(num_atom - 1)
        for s in sub:
            result.append(s + [False])
            result.append(s + [True])
        return result

def run_logic(p, assignment):
    num_atom = len(assignment)
    for a in range(num_atom):
        if not p[0][a]:
            continue
        if assignment[a] and not p[1][a]:
            return True
        if not assignment[a] and p[1][a]:
            return True
    return False
    

def logic_checker(p1, p2, pc):
    '''
    check not p1 or not p2 or pc is always True
    '''
    if p1[0][0] == False and \
        p1[0][1] == False and \
        p1[0][2] and \
        not p1[1][2] and \
        p2[0][0] == False and \
        p2[0][1] == False and \
        p2[0][2] and \
        p2[1][2] and \
        pc[0][0] == False and \
        pc[0][1] and \
        pc[0][2] == False and \
        pc[1][2] == False:
        import ipdb;ipdb.set_trace(context=15)
    num_atom = len(p1[0])
    assignments = list_bool_vector(num_atom)
    for assignment in assignments:
        is_true = False
        if not run_logic(p1, assignment) or \
                not run_logic(p2, assignment) or \
                run_logic(pc, assignment):
            is_true = True
        if not is_true:
            return False
    return True

def cartesian_list(select_atom, negation):
    result = []
    for r1 in select_atom:
        for r2 in negation:
            if any(a == False and b == True for a , b in zip(r1, r2)):
                continue
            result.append((r1, r2))
    return result

def convert_logic(p1, p2, pc):
    def convert_one(p):
        all_comp = []
        for i, x in enumerate(p[0]):
            if not x:
                continue
            c = chr(ord('X') + i)
            if p[1][i]:
                all_comp.append('not {}'.format(c))
            else:
                all_comp.append(c)
        return ' or '.join(all_comp)
    s1 = convert_one(p1)
    s2 = convert_one(p2)
    sc = convert_one(pc)
    return s1, s2, sc 

def logic_task():
    true_pairs, false_pairs = generate_true_false()
    readable_true_logic = [convert_logic(*p) for p in true_pairs]
    readable_false_logic = [convert_logic(*p) for p in false_pairs]
    tsv_writer(readable_true_logic, '/home/jianfw/work/q/true_logic.tsv')
    tsv_writer(readable_false_logic, '/home/jianfw/work/q/false_logic.tsv')

def generate_true_false():
    # enumerate all logic paires
    num_premise = 2
    assert num_premise == 2
    num_atom = 3
    select_atom = list_bool_vector(num_atom)
    negation = list_bool_vector(num_atom)
    prop = cartesian_list(select_atom, negation)
    true_pairs = []
    false_pairs = []
    for p1 in prop:
        if all(not x for x in p1[0]):
            continue
        for p2 in prop:
            if all(not x for x in p2[0]):
                continue
            if all(v1 == v2 for a1, a2 in zip(p1, p2) for v1, v2 in zip(a1,
                a2)):
                continue
            for pc in prop:
                if all(not x for x in pc[0]):
                    continue
                if all(v1 == v2 for a1, a2 in zip(p1, pc) for v1, v2 in zip(a1,
                    a2)):
                    continue
                if all(v1 == v2 for a1, a2 in zip(pc, p2) for v1, v2 in zip(a1,
                    a2)):
                    continue
                #if p1[0][0] and \
                        #not p1[0][1] and \
                        #not p1[0][2] and \
                        #not p1[1][0] and \
                        #p2[0][0] and \
                        #not p2[0][1] and \
                        #not p2[0][2] and \
                        #not p2[1][0] and \
                        #pc[0][0] and \
                        #not pc[0][1] and \
                        #not pc[0][2] and \
                        #not pc[1][0]:
                    #import ipdb;ipdb.set_trace(context=15)
                is_always_true = logic_checker(p1, p2, pc)
                if is_always_true:
                    true_pairs.append((p1, p2, pc))
                else:
                    false_pairs.append((p1, p2, pc))
    return true_pairs, false_pairs
    

def test_merge_prediction_to_gt():
    from yolotrain import CaffeWrapper
    test_data = 'MSLogoClean'

    datas=[
        'imagenet1kLocClean',
        'mturk700_url_as_keyClean',
        'crawl_office_v1',
        'crawl_office_v2',
        'Materialist',
        'VisualGenomeClean',
        'Naturalist',
        '4000_Full_setClean',
        'clothingClean',
        'open_images_clean_1',
        'open_images_clean_2',
        'open_images_clean_3',
        ]
    for test_data in datas:
        for test_split in ['train', 'test', 'trainval']:
            full_expid = 'voc0712_darknet19_448_B'
            c = CaffeWrapper(full_expid=full_expid, 
                    load_parameter=True,
                    test_data=test_data,
                    test_split=test_split)
            m = c.best_model()
            prediction = c._predict_file(m)
            if not op.isfile(prediction):
                logging.info('no {}'.format(prediction))
                continue
            dataset = TSVDataset(test_data)
            gt = dataset.get_data(test_split, 'label', version=-1)
            v = dataset.get_latest_version(test_split, 'label')
            out_file = dataset.get_data(test_split, 'label', v + 1)
            merge_prediction_to_gt(prediction, gt, out_file)

    for test_data in datas:
        populate_dataset_details(test_data)

def merge_similar_boundingbox(data):
    dataset = TSVDataset(data)
    split = 'trainval'
    t = 'label'
    label_src_rows = dataset.iter_data(split, t, -1)
    image_src_rows = dataset.iter_data(split)
    def gen_rows():
        for i, label_row in enumerate(label_src_rows):
            if (i % 10000) == 0:
                logging.info(i)
            src_rects = json.loads(label_row[1])
            rects = copy.deepcopy(src_rects)
            all_group = []
            while len(rects) > 0:
                curr_rect = rects[-1]
                del rects[-1]
                found = None
                for g in all_group:
                    for r in g:
                        if r['class'] == curr_rect['class']:
                            iou = calculate_iou(r['rect'], curr_rect['rect'])
                            if iou > 0.6:
                                found = g
                                break
                    if found is not None:
                        break
                if found is not None:
                    found.append(curr_rect)
                else:
                    all_group.append([curr_rect])
            dest_rects = []
            for g in all_group:
                rect = {'class': g[0]['class']}
                x0, y0, x1, y1 = g[0]['rect']
                for r in g:
                    x0 = max(r['rect'][0], x0)
                    y0 = max(r['rect'][1], y0)
                    x1 = min(r['rect'][2], x1)
                    y1 = min(r['rect'][3], y1)
                rect['rect'] = [x0, y0, x1, y1]
                rect['merge_from'] = g
                dest_rects.append(rect)
            yield label_row[0], json.dumps(dest_rects)
    v = dataset.get_latest_version(split, t)
    dataset.write_data(gen_rows(), split, t, v + 1)

def netprototxt_to_netspec():
    prototxt = '/home/jianfw/code/SENet/models/SE-BN-Inception.prototxt'
    net = load_net(prototxt)
    all_line = []
    # topname = the top value in the proto
    # effectivetopname = the name stored in self.n.tops, which might not be
    # equal to topname if it is an in-place operation, e.g. relu
    topname_to_effectivetopname = {}
    def get_set(topname):
        if topname not in topname_to_effectivetopname:
            topname_to_effectivetopname[topname] = topname
        return topname_to_effectivetopname[topname]
    for l in net.layer:
        def update_bottom():
            if len(l.bottom) == 1 and len(l.bottom[0]) > 0:
                line = "self.set_bottom('{}')".format(get_set(l.bottom[0]))
                # make sure the bottom is correct
                all_line.append(line)
        update_bottom()
        extra_param = None
        if l.type == 'Convolution':
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            nout = l.convolution_param.num_output
            line = 'self.conv(nout={}'.format(nout)

            ks = l.convolution_param.kernel_size
            assert len(ks) == 1
            ks = ks[0]
            line = '{}, ks={}'.format(line, ks)
            stride = l.convolution_param.stride
            if len(stride) == 1:
                stride = stride[0]
                line = '{}, stride={}'.format(line, stride)
            else:
                assert len(stride) == 0
            bias = l.convolution_param.bias_term
            line = '{}, bias={}'.format(line, bias)
            pad = l.convolution_param.pad
            if len(pad) == 1:
                line = '{}, pad={}'.format(line, pad[0])
            else:
                assert len(pad) == 0
        elif l.type == 'BatchNorm':
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            extra_param = l.batch_norm_param
            line = "self.bn("
        elif l.type == 'Scale':
            assert len(l.top) == 1
            if len(l.bottom) == 1:
                line = "self.scale(bias={}".format(l.scale_param.bias_term)
            else:
                assert len(l.bottom) > 0
                line = "self.scale_m({}".format(', '.join(["'{}'".format(
                    get_set(b)) for b in l.bottom]))
                line = "{}, bias={}".format(line, l.scale_param.bias_term)
            line = "{}, axis={}".format(line, l.scale_param.axis)
        elif l.type == 'Pooling':
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            if l.pooling_param.pool == 0:
                if l.pooling_param.global_pooling:
                    line = 'self.maxpoolglobal('
                else:
                    line = 'self.maxpool('
            elif l.pooling_param.pool == 1:
                if l.pooling_param.global_pooling:
                    line = 'self.avepoolglobal('
                else:
                    line = 'self.avepool('
            else:
                assert False
            extra_param = l.pooling_param
        elif l.type == 'Input':
            continue
        elif l.type == 'ReLU':
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            line = "self.relu("
        elif l.type == 'Concat':
            assert len(l.top) == 1
            line = 'self.set_bottom(None)'
            all_line.append(line)
            line = "self.concat_m("
            for i, b in enumerate(l.bottom):
                if i == 0:
                    line = "{}'{}'".format(line, b)
                else:
                    line = "{}, '{}'".format(line, b)
        elif l.type == 'Sigmoid':
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            line = "self.sigmoid("
        elif l.type == 'Reshape':
            assert len(l.bottom) == 1
            assert len(l.top) == 1
            line = "self.reshape(shape={'dim': ["
            for i, d in enumerate(l.reshape_param.shape.dim):
                if i == 0:
                    line = "{}{}".format(line, d)
                else:
                    line = '{}, {}'.format(line, d)
            line = '{}]}}'.format(line)
        elif l.type == 'InnerProduct':
            line = "self.fc(nout=1000".format(l.name)
        elif l.type == 'Softmax':
            continue
        else:
            import ipdb;ipdb.set_trace(context=15)
        if not line.endswith('('):
            line = "{}, ".format(line)
        if extra_param:
            all_field = extra_param.ListFields()
            for key, value in all_field:
                name = key.name
                if l.type == 'BatchNorm' and key.name == 'use_global_stats':
                    continue
                if l.type == 'Pooling':
                    if key.name == 'pool' or key.name == 'global_pooling':
                        # this is deployment specific 
                        continue
                if l.type == 'Pooling' and key.name == 'kernel_size':
                    name = 'ks'
                if type(value) == int or \
                        type(value) == bool or \
                        type(value) == str:
                    surround_value = '' if type(value) is not str else "'"
                    line = "{0}{1}={3}{2}{3}, ".format(
                            line, name, value, surround_value)
                else:
                    assert False
        assert len(l.top) == 1
        if l.top[0] not in l.bottom:
            # topname is not equal to bottom name, we need to specify the
            # top name = effective top name
            line = "{}layername='{}', in_place=False, ".format(line, 
                    l.top[0])
        else:
            # topname is equal to bottom name
            effectivetopname = l.name + '$' + l.top[0]
            line = "{}layername='{}', in_place=True, ".format(line, effectivetopname)
            topname_to_effectivetopname[l.top[0]] = effectivetopname
        line = "{}name='{}'".format(line, l.name)
        line = '{})'.format(line)
        all_line.append(line)
        #line = 'self.n.to_proto()'
        #all_line.append(line)

    logging.info('\n'.join(all_line))

def create_voc_person():
    source_dataset = TSVDataset('voc20')
    dest_dataset = TSVDataset('voc20_person')
    rows = tsv_reader(source_dataset.get_data('test'))

    def gen_rows():
        for row in rows:
            rects= json.loads(row[1])
            rects = [r for r in rects if r['class'] == 'person']
            #for r in rects:
                #r['class'] = 'Person'
            yield row[0], json.dumps(rects), row[2]

    tsv_writer(gen_rows(), dest_dataset.get_data('test'))

def sync_global_data():
    while True:
        all_data = os.listdir('./data')
        for data in all_data:
            logging.info(data)
            philly_upload_dir('data/{}'.format(data), 
                    'jianfw/data/qd_data/',
                    vc='input')
        time.sleep(60)

def merge_prediction_to_gt(prediction_file, 
        gt_file, out_file, srclabel_to_destlabel=None):
    prob_th = 0.7
    iou_th = 0.3
    key_to_pred = {}
    prediction = tsv_reader(prediction_file)
    for p in prediction:
        assert len(p) == 2
        key = p[0]
        rects = json.loads(p[1])
        key_to_pred[key] = rects
    gts = tsv_reader(gt_file)
    def gen_rows():
        num_image = 0
        num_added = 0
        for i, gt in enumerate(gts):
            if (i % 1000) == 0:
                logging.info('{} - {}'.format(prediction_file, i))
            num_image = num_image + 1
            key = gt[0]
            g_rects = json.loads(gt[1])
            p_rects = key_to_pred[key]
            p_rects = [r for r in p_rects if r['conf'] > prob_th]
            if srclabel_to_destlabel:
                p_rects = [r for r in p_rects if r['class'] in srclabel_to_destlabel]
                for r in p_rects:
                    r['class'] = srclabel_to_destlabel[r['class']]
            added = []
            for p_rect in p_rects:
                ious = [calculate_iou(p_rect['rect'], g_rect['rect']) 
                        for g_rect in g_rects]
                if max(ious) < iou_th:
                    num_added = num_added + 1
                    p_rect['from'] = prediction_file
                    # add p_rect
                    added.append(p_rect)
            g_rects.extend(added)
            yield key, json.dumps(g_rects)
        logging.info(num_image)
        logging.info(num_added)
    tsv_writer(gen_rows(), out_file)

def destroy_label_field(data, split):
    dataset = TSVDataset(data)
    src_file = dataset.get_data(split)
    dest_file = src_file + '.remove.col.tsv'
    rows = dataset.iter_data(split)
    def gen_rows():
        for row in rows:
            row[1] = 'd'
            yield row
    tsv_writer(gen_rows(), dest_file)
    src_idx = '{}.lineidx'.format(op.splitext(src_file)[0])
    os.remove(src_file)
    os.remove(src_idx)
    os.rename(dest_file, src_file)
    dest_idx = '{}.lineidx'.format(op.splitext(dest_file)[0])
    os.rename(dest_idx, src_idx)

def convertcomposite_to_standard(src_data, 
        dest_data, split='train'):
    dataset = TSVDataset(src_data)
    out_dataset = TSVDataset(dest_data)
    rows = dataset.iter_data(split)
    tsv_writer(rows, out_dataset.get_data(split))

def get_one_row(source_name, source_label, cache):
    if source_name not in cache['name_to_dataset']:
        logging.info('constructing dataset {}'.format(source_name))
        cache['name_to_dataset'][source_name] = TSVDataset(source_name)
    dataset = cache['name_to_dataset'][source_name]
    if source_name not in cache['name_to_split']:
        splits = ['train', 'trainval', 'test']
        ss = None
        for s in splits:
            if op.isfile(dataset.get_data(s)):
                ss = s
                break
        assert ss is not None
        logging.info('cache {} -> {}'.format(source_name, ss))
        cache['name_to_split'][source_name] = ss
    split = cache['name_to_split'][source_name]
    if source_name not in cache['name_to_inverted_list']:
        inverted = dataset.load_inverted_label(split)
        logging.info('cache inverted index: {}'.format(source_name))
        cache['name_to_inverted_list'][source_name] = inverted
    inverted = cache['name_to_inverted_list'][source_name]
    idx = inverted[source_label]
    if source_name not in cache['name_to_tsvfile']:
        logging.info('construct tsv interface {} - {}'.format(source_name, split))
        cache['name_to_tsvfile'][source_name] = TSVFile(dataset.get_data(split))
    tsv = cache['name_to_tsvfile'][source_name]
    assert len(idx) > 0
    return tsv.seek(idx[0])

def gen_html_label_mapping(data):
    #data = 'Tax4k_V1_6'
    #data_source_names = ['coco2017',
                #'voc0712', 
                #'brand1048Clean',
                #'imagenet3k_448',
                #'imagenet22k_448',
                #'imagenet1kLoc',
                #'mturk700_url_as_key',
                #'crawl_office_v1',
                #'crawl_office_v2',
                #'Materialist',
                #'VisualGenomeClean',
                #'Naturalist',
                #'4000_Full_set',
                #'MSLogo',
                #'clothing',
                #'open_images']
    sub_folder = 'category_viewer'
    dataset = TSVDataset(data)
    param = load_from_yaml_file(op.join(dataset._data_root,
        'generate_parameters.yaml'))
    data_source_names = param[1]['datas']
    file_name = op.join(dataset._data_root, 'root.yaml')
    with open(file_name, 'r') as fp:
        config_tax = yaml.load(fp)
    result = []
    tax = Taxonomy(config_tax)
    cache = {'name_to_dataset': {},
            'name_to_split': {},
            'name_to_inverted_list': {},
            'name_to_tsvfile': {}}
    need_one = True
    need_two = True

    for k, node in enumerate(tax.root.iter_search_nodes()):
        if node is tax.root:
            continue
        curr_row = {}
        if (k % 100) == 0:
            logging.info(k)
        #if k > 5:
            #break
        for source_name in data_source_names:
            if not hasattr(node, source_name):
                continue
            values = node.__getattribute__(source_name)
            if values is None:
                continue
            source_labels = [s.strip() for s in
                values.split(',')]
            curr_row[source_name] = []
            for source_label in source_labels:
                row = get_one_row(source_name, source_label, cache)
                rects = [r for r in json.loads(row[1]) if r['class'] == source_label]
                assert len(rects)
                im = img_from_base64(row[-1])
                draw_bb(im, [r['rect'] for r in rects], [r['class'] for r in rects])
                rel_path = op.join(source_name, '{}_{}.jpg'.format(hash(row[0]),
                    source_label)).replace(' ', '')
                dest_file = op.join(dataset._data_root, sub_folder,
                        rel_path)
                save_image(im, dest_file)
                readable_name = '{}({})'.format(source_label,
                    get_nick_name(noffset_to_synset(source_label))) if is_noffset(source_label) else source_label
                readable_name = '{}: {}'.format(source_name, readable_name)
                curr_row[source_name].append((readable_name, rel_path))
        result.append((node.name, curr_row))

    # write to html
    from jinja2 import Environment, FileSystemLoader
    j2_env = Environment(loader=FileSystemLoader('./'), trim_blocks=True)
    r = j2_env.get_template('aux_data/html_template/composite_dataset_category_viewer2.html').render(
        source_names=data_source_names,
        term_infos=result)
    write_to_file(r, op.join(dataset._data_root, sub_folder, 'index.html'))

def calc_map_per_class(prediction, gt):
    '''
    prediction is a matrix, N*K, N is the number of samples; K is the number of
    features
    gt: N. the label idx of that row
    '''
    N, K = prediction.shape
    result = []
    for k in range(K):
        curr_p = prediction[:, k]
        curr_g = [g == k for g in gt]
        idx = sorted(range(N), key=lambda k: -curr_p[k])
        matches = [curr_g[i] for i in idx]
        ap = calculate_ap_by_true_list(matches, np.sum(curr_g))
        result.append(ap)

    return result

def per_class_check():
    folder = \
        'output/Tax1300SGV1_1_darknet19_448_B_noreorg_extraconv2_tree_init3491_IndexLossWeight0_bb_nobb/'
    map_per_class_file = op.join(folder, 'snapshot',
        'model_iter_244004.caffemodel.Tax1300SGV1_1_with_bb.maintainRatio.OutTreePath.report.class_ap.json')
    per_class_map = json.loads(read_to_buffer(map_per_class_file))
    import ipdb;ipdb.set_trace(context=15)


def update_config():
    all_file, all_folder = parse_philly_ls_output(philly_ls('jianfw/code', vc='input',
        return_output=True))
    if any('run.py' in f for f in all_file):
        philly_remove('jianfw/code/run.py')
    philly_upload('philly/run.py', 'jianfw/code/', vc='input')

def parse_seq(path):
    p = op.basename(path)
    n = op.splitext(p)[0]
    return int(float(n))

def create_vot_dataset():
    root_folder = '/home/jianfw/code/GOTURN/data/vot'
    all_folder = os.listdir(root_folder)
    for folder in all_folder:
        image_folder = op.join(root_folder, folder)
        if not op.isdir(image_folder):
            continue
        all_image = glob.glob(op.join(image_folder, '*.jpg'))
        all_seq = [parse_seq(image) for image in all_image]
        image_seqs = sorted(zip(all_image, all_seq), key=lambda x: x[1])
        if image_seqs[0][1] == 0:
            image_seqs = image_seqs[1:]
        assert image_seqs[0][1] == 1
        gt_file = op.join(image_folder, 'groundtruth.txt')
        gts = load_list_file(gt_file)
        rects = []
        for gt in gts:
            ax, ay, bx, by, cx, cy, dx, dy = [float(g) for g in gt.split(',')]
            x0 = min([ax, bx, cx, dx]) - 1
            y0 = min([ay, by, cy, dy]) - 1
            x1 = max([ax, bx, cx, dx]) - 1
            y1 = max([ay, by, cy, dy]) - 1
            rects.append([x0, y0, x1, y1])
        dataset_name = 'vot_' + folder
        dataset = TSVDataset(dataset_name)
        def gen_rows(image_seqs, rects, idx_start, idx_end):
            for i in range(idx_start, idx_end):
                r = rects[i]
                image_path = image_seqs[i][0]
                assert len(r) == 4
                rect = [{'class': folder, 'rect': r}]
                yield op.basename(image_path), json.dumps(rect), base64.b64encode(read_to_buffer(image_path))
        tsv_writer(gen_rows(image_seqs, rects, 0, 1), dataset.get_data('train'))
        tsv_writer(gen_rows(image_seqs, rects, 1, len(image_seqs)),
                dataset.get_data('test'))
        populate_dataset_details(dataset_name)

def inverse_sigmoid(y):
    '''
    y = 1/(1 + exp(-x))
    '''
    return -np.log(1. / y - 1.)

def l2minimized(xs, ys, regularizer=0.01, log=False):
    X1 = np.vstack(xs)
    X = np.hstack((X1, np.ones((X1.shape[0], 1))))
    XTX = np.dot(X.transpose(), X)
    left = XTX  + regularizer * np.identity(X.shape[1])
    Y = np.vstack(ys)
    right = np.dot(X.transpose(), Y)
    W = la.solve(left, right)
    if log:
        values, _ = la.eig(XTX)
        values = sorted(values)
        logging.info('{}'.format(','.join(map(str, values[-10:]))))
        logging.info('{}'.format(','.join(map(str, X.shape))))
    return W[:-1], W[-1]

def test_init_last_conv_by_min_l2():
    expid = 'B_noreorg_extraconv2_maxIter.0_TsvBoxSamples50'
    full_expid = 'vot_ball_darknet19_448_B_noreorg_extraconv2_maxIter.0_TsvBoxSamples50'
    pretrained_proto = 'output/imagenet200_darknet19_448_B_noreorg_extraconv2/train.prototxt'
    pretrained_model = 'output/imagenet200_darknet19_448_B_noreorg_extraconv2/snapshot/model_iter_913134.caffemodel'
    new_proto = 'output/{}/train.prototxt'.format(full_expid)
    new_model = 'output/{}/snapshot/model_iter_0.caffemodel'.format(full_expid) # this is actually the output
    #process_run(init_last_conv_by_min_l2, pretrained_model, pretrained_model, new_proto,
        #new_model)
    init_last_conv_by_min_l2(pretrained_proto, pretrained_model, new_proto, new_model)
    return
    from yolotrain import CaffeWrapper
    test_on_train = False
    c = CaffeWrapper(data='vot_ball', net='darknet19_448', 
            expid=expid,
            load_parameter=True, 
            gpus=[0],
            test_on_train=test_on_train)
    m = c.best_model()
    m.model_iter = 0
    m.model_param = new_model
    predict_result = c.predict(m)
    c.evaluate(m, predict_result)

def init_last_conv_by_min_l2(pretrained_proto, pretrained_model, new_proto, new_model):
    # add the gt_target blob into the train proto
    pnet = load_net(new_proto)
    for l in pnet.layer:
        if l.type == 'RegionTarget':
            l.top.append('gt_target')
        if l.type == 'BatchNorm':
            l.batch_norm_param.use_global_stats = True
        if l.type == 'TsvBoxData':
            l.box_data_param[0].random_scale_max = 8
    regularizer_wh = 100
    regularizer_xy = 10
    regularizer_obj = 1

    modified_new_proto = '/tmp/train.prototxt'
    write_to_file(str(pnet), modified_new_proto)

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(str(modified_new_proto), 
            caffe.TRAIN)
    net.copy_from(pretrained_model, ignore_shape_mismatch=True)
    feature_name, t_xy_name, t_wh_name = 'extra_conv20', 't_xy', 't_wh'
    gt_target_name, label_name, t_o_obj_name = 'gt_target', 'label', 't_o_obj'
    t_o_noobj_name = 't_o_noobj'
    
    XY_x_anchor = None
    total_image = 10
    start_time = time.time()
    obj_norm = 1
    share_xy = True
    while True:
        net.forward()
        # xy
        feature = net.blobs[feature_name].data
        t_xy = net.blobs[t_xy_name].data
        t_wh = net.blobs[t_wh_name].data
        gt_target = net.blobs[gt_target_name].data
        label = net.blobs[label_name].data
        t_o_obj = net.blobs[t_o_obj_name].data
        # this target has 0 and the inverse of sigmoid is -infinity
        #t_o_noobj = net.blobs[t_o_noobj_name].data

        num_image = gt_target.shape[0]
        num_max_gt = gt_target.shape[1]
        # the first is Xs, while the second is the Ys
        xy = ([], [])
        Xs = []
        num_anchor = t_xy.shape[1] / 2
        if XY_x_anchor is None:
            feature_width = t_xy.shape[3]
            feature_height = t_xy.shape[2]
            XY_x_anchor = [] 
            XY_y_anchor = [] 
            XY_w_anchor = [] 
            XY_h_anchor = [] 
            # do not use [([], [])] * num_anchor since all the list will be the
            # same for different num_anchor
            for a in range(num_anchor):
                XY_x_anchor.append(([], []))
                XY_y_anchor.append(([], []))
                XY_w_anchor.append(([], []))
                XY_h_anchor.append(([], []))
            XY_obj_anchor = ([], [])
        t_o_noobj = np.zeros((num_image, num_anchor, feature_height,
            feature_width))
        t_o_noobj[:] = -obj_norm
        for b in range(num_image):
            for g in range(num_max_gt):
                target_x, target_y, target_n = [int(x) for x in
                    gt_target[b, g, :, 0]]
                if target_x < 0 or target_y < 0 or target_n < 0:
                    assert target_x < 0 and target_y < 0 and target_n < 0
                    continue
                # append the Xs
                f = feature[b, :, target_y, target_x]

                # Y for xy
                curr_t_x = t_xy[b, target_n, target_y, target_x]
                lx = label[b, g * 5]
                assert np.abs(curr_t_x - lx * feature_width + np.floor(lx *
                    feature_width)) < 0.0001
                curr_t_y = t_xy[b, target_n + num_anchor, target_y, target_x]
                ly = label[b, g * 5 + 1]
                assert np.abs(curr_t_y - ly * feature_height + np.floor(ly *
                        feature_height)) < 0.0001
                if share_xy:
                    for n in range(num_anchor):
                        XY_x_anchor[n][0].append(f)
                        XY_x_anchor[n][1].append(inverse_sigmoid(curr_t_x))
                        XY_y_anchor[n][0].append(f)
                        XY_y_anchor[n][1].append(inverse_sigmoid(curr_t_y))
                else:
                    XY_x_anchor[target_n][0].append(f)
                    XY_x_anchor[target_n][1].append(inverse_sigmoid(curr_t_x))
                    XY_y_anchor[target_n][0].append(f)
                    XY_y_anchor[target_n][1].append(inverse_sigmoid(curr_t_y))

                # t_w
                curr_t_w = t_wh[b, target_n, target_y, target_x]
                XY_w_anchor[target_n][0].append(f)
                XY_w_anchor[target_n][1].append(curr_t_w)
                
                # t_h
                curr_t_h = t_wh[b, target_n + num_anchor, target_y, target_x]
                XY_h_anchor[target_n][0].append(f)
                XY_h_anchor[target_n][1].append(curr_t_h)

                t_o_noobj[b, target_n, target_y, target_x] = obj_norm

        f = feature.transpose((0, 2, 3, 1)) 
        f = f.reshape((-1, f.shape[-1]))
        XY_obj_anchor[0].append(f)
        f = t_o_noobj.transpose((0, 2, 3, 1))
        f = f.reshape((-1, f.shape[-1]))
        XY_obj_anchor[1].append(f)
        total_image = total_image - num_image
        if total_image <= 0:
            break
    
    # xywh
    last_conv_param = net.params['last_conv']
    for target_n in range(num_anchor):
        if len(XY_x_anchor[target_n][0]) > 0:
            logging.info('x - {} - {}'.format(target_n,
                len(XY_x_anchor[target_n][0])))
            w, b = l2minimized(*XY_x_anchor[target_n],
                    regularizer=regularizer_wh) 
            # print the norm comparision
            old_w_norm = np.sum(last_conv_param[0].data[target_n, :, :, :][:] * \
                    last_conv_param[0].data[target_n, :, :, :][:])
            new_w_norm = np.sum(w[:] * w[:])
            old_b = last_conv_param[1].data[target_n]
            new_b = b
            logging.info('{}-w: old({}); new({})'.format(target_n, old_w_norm, 
                new_w_norm))
            logging.info('{}-b: old({}); new({})'.format(target_n, old_b, 
                new_b))

            last_conv_param[0].data[target_n, :, :, :] = w[:, :,
                    np.newaxis]
            last_conv_param[1].data[target_n] = b 
        else:
            logging.info('empty data for x - {}'.format(target_n))
        if len(XY_y_anchor[target_n][0]) > 0:
            logging.info('y - {} - {}'.format(target_n,
                len(XY_y_anchor[target_n][0])))
            w, b = l2minimized(*XY_y_anchor[target_n],
                    regularizer=regularizer_wh) 

            # print the norm comparision
            old_w_norm = np.sum(last_conv_param[0].data[target_n + num_anchor, :, :, :][:] * \
                    last_conv_param[0].data[target_n + num_anchor, :, :, :][:])
            new_w_norm = np.sum(w[:] * w[:])
            old_b = last_conv_param[1].data[target_n + num_anchor]
            new_b = b
            logging.info('{}-w: old({}); new({})'.format(target_n, old_w_norm, 
                new_w_norm))
            logging.info('{}-b: old({}); new({})'.format(target_n, old_b, 
                new_b))

            last_conv_param[0].data[target_n + num_anchor, :, :] = w[:,
                    :, np.newaxis]
            last_conv_param[1].data[target_n + num_anchor] = b 
        else:
            logging.info('empty data for y - {}'.format(target_n))
        if len(XY_w_anchor[target_n][0]) > 0:
            logging.info('w - {} - {}'.format(target_n,
                len(XY_w_anchor[target_n][0])))
            w, b = l2minimized(*XY_w_anchor[target_n], 
                    regularizer=regularizer_wh) 

            # print the norm comparision
            old_w_norm = np.sum(last_conv_param[0].data[target_n + 2 * num_anchor, :, :, :][:] * \
                    last_conv_param[0].data[target_n + 2 * num_anchor, :, :, :][:])
            new_w_norm = np.sum(w[:] * w[:])
            old_b = last_conv_param[1].data[target_n + 2 * num_anchor]
            new_b = b
            logging.info('{}-w: old({}); new({})'.format(target_n, old_w_norm, 
                new_w_norm))
            logging.info('{}-b: old({}); new({})'.format(target_n, old_b, 
                new_b))

            last_conv_param[0].data[target_n + 2 * num_anchor, :, :] = w[:, :,
                    np.newaxis]
            last_conv_param[1].data[target_n + 2 * num_anchor] = b 
        else:
            logging.info('empty data for w - {}'.format(target_n))
        if len(XY_h_anchor[target_n][0]) > 0:
            logging.info('h - {} - {}'.format(target_n,
                len(XY_h_anchor[target_n][0])))
            w, b = l2minimized(*XY_h_anchor[target_n], 
                    regularizer=regularizer_wh) 

            # print the norm comparision
            old_w_norm = np.sum(last_conv_param[0].data[target_n + 3 * num_anchor, :, :, :][:] * \
                    last_conv_param[0].data[target_n + 3 * num_anchor, :, :, :][:])
            new_w_norm = np.sum(w[:] * w[:])
            old_b = last_conv_param[1].data[target_n + 3 * num_anchor]
            new_b = b
            logging.info('{}-w: old({}); new({})'.format(target_n, old_w_norm, 
                new_w_norm))
            logging.info('{}-b: old({}); new({})'.format(target_n, old_b, 
                new_b))

            last_conv_param[0].data[target_n + 3 * num_anchor, :, :] = w[:, :,
                    np.newaxis]
            last_conv_param[1].data[target_n + 3 * num_anchor] = b 
        else:
            logging.info('empty data for h - {}'.format(target_n))
    
    W, b = l2minimized(*XY_obj_anchor, regularizer=regularizer_obj, 
            log=True)

    # print the norm comparision
    old_w = last_conv_param[0].data[4 * num_anchor : (5 * num_anchor), :, :]
    old_w_norm = np.sum(old_w[:] * old_w[:])
    new_w_norm = np.sum(W[:] * W[:])
    old_b = last_conv_param[1].data[4 * num_anchor : (5 * num_anchor)]
    old_b = np.sum(old_b * old_b)
    new_b = b * b
    logging.info('{}-w: old({}); new({})'.format(target_n, old_w_norm, 
        new_w_norm))
    logging.info('{}-b: old({}); new({})'.format(target_n, old_b, 
        new_b))

    last_conv_param[0].data[4 * num_anchor : (5 * num_anchor), :, :] = \
            W.transpose()[:, :, np.newaxis, np.newaxis]
    last_conv_param[1].data[4 * num_anchor : (5 * num_anchor)] = \
            b
    elapsed = time.time() - start_time
    
    # visualize the results
    visualization = True
    if visualization:
        net.forward(start='last_conv')
        obj = net.blobs['obj'].data
        gt_target = net.blobs[gt_target_name].data
        all_image = network_input_to_image(net.blobs['data'].data, [104, 117, 123])
        for n in range(num_image):
            all_im = []
            _, ax = plt.subplots(2, num_anchor + 1)
            ax[0, 0].imshow(all_image[n])
            all_output_obj = []
            for a in range(num_anchor):
                im = obj[n, a, :, :]
                all_output_obj.extend(im.flatten())
                #im = 255. * (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
                im = 255. * im
                im = im.astype(np.uint8)
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
                ax[0, a + 1].imshow(im)
            all_gt_im = []
            for a in range(num_anchor):
                im = np.zeros((feature_height, feature_width, 3), dtype=np.uint8)
                all_gt_im.append(im)
            for g in range(num_max_gt):
                target_x, target_y, target_n = [int(x) for x in
                    gt_target[n, g, :, 0]]
                if target_x < 0 or target_y < 0 or target_n < 0:
                    continue
                all_gt_im[target_n][target_y, target_x, :] = 255
            ax[1, 0].set_yscale('log', nonposy='clip')
            ax[1, 0].hist(all_output_obj, 50)
            for i, im in enumerate(all_gt_im):
                ax[1, 1 + i].imshow(im)
            plt.show()
            plt.close()

    logging.info('time cost: {}'.format(elapsed))
    net.save(new_model)

def count_num_alov300():
    root_folder = '/home/jianfw/code/GOTURN/data/alov300/imagedata++'
    t = 0
    all_folder = os.listdir(root_folder)
    for folder in all_folder:
        sub_folders = os.listdir(op.join(root_folder, folder))
        t = t + len(sub_folders)
    logging.info(t)

def study_imagenet3k():
    # visualize_3k()
    # check if exists
    mapper = LabelToSynset()
    ss = mapper.convert('cup')
    tree = build_imagenet3k()
    result = tree.search_nodes(noffset=synset_to_noffset(ss))
    for leaf in result[0].iter_leaves():
        logging.info(leaf.noffset)
        ensue_untar(leaf.noffset)

def ensure_untar(noffset, tar_folder='/vighd/data/ImageNet22K',
        target_root_folder = '/raid/jianfw/work/ImageNet22K'):
    
    target_folder = op.join(target_root_folder, noffset)
    if op.isdir(target_folder):
        return True
    tar_file = op.join(tar_folder, noffset + '.tar')
    if not op.isfile(tar_file):
        return False
    t = tarfile.open(tar_file)
    t.extractall(target_folder)
    return True

def get_expansion_list():
    expand_list = ['dog', 'cat', 'tiger',
            'snake', 'frog', 'butterfly', 'worm',
            'bird', 'fish', 'dinosaur', 'weapon', 
            'airplane', 'car', 'train', 'ship', 'apple']
    return expand_list

def bing2k():
    j_cat = '/vighd/dlworkspace/work/jianfw/data/bing2k/Jian1570A2_combined_categories.tsv'
    j_map = '/vighd/dlworkspace/work/jianfw/data/bing2k/Jian1570A2_combined_ImageNetSynsetExpension.tsv'

    rows = tsv_reader(j_cat)
    labels = []
    for row in rows:
        idx = float(row[0])
        name = row[1]
        assert len(labels) == idx
        labels.append(name)
    
    label_to_noffsets = {}
    shortname_to_fullname = {}
    rows = tsv_reader(j_map)
    for row in rows:
        curr_label = row[0].split(',')[-1]
        if curr_label not in shortname_to_fullname:
            shortname_to_fullname[curr_label] = row[0]
        else:
            assert row[0] == shortname_to_fullname[curr_label]
        noffset = row[1]
        if curr_label not in label_to_noffsets:
            label_to_noffsets[curr_label] = [noffset]
        else:
            label_to_noffsets[curr_label] += [noffset]
    for l in labels:
        if l not in label_to_noffsets:
            logging.info(l)


def visualize_3k():
    # visualize the images
    image_root = '/raid/jianfw/work/ImageNet22K'
    annotation_root = '/home/jianfw/work/imagenet3k/annotation/Annotation/'
    folders = glob.glob(annotation_root + '*')
    for curr_folder in folders:
        noffset = op.basename(curr_folder)
        files = glob.glob(op.join(curr_folder, '*.xml'))
        ensure_untar(noffset)
        for curr_file in files: 
            xml_basename = op.splitext(op.basename(curr_file))[0]
            gt, height, width = load_voc_xml(curr_file)
            image_file_name = op.join(image_root, noffset,
                xml_basename + ".JPEG")
            if not op.isfile(image_file_name):
                logging.info('not exist: {}'.format(image_file_name))
                continue
            im = cv2.imread(image_file_name, cv2.CV_LOAD_IMAGE_COLOR)

            draw_bb(im, [g['rect'] for g in gt], [g['class'] for g in gt])
            show_image(im)


def visualize_noffset_in_3k(noffset, out_folder):
    image_root = '/raid/jianfw/work/ImageNet22K'
    annotation_root = '/home/jianfw/work/imagenet3k/annotation/Annotation/'
    ensure_untar(noffset)
    curr_folder = op.join(annotation_root, noffset)
    files = glob.glob(op.join(curr_folder, '*.xml'))
    for curr_file in files:
        xml_basename = op.splitext(op.basename(curr_file))[0]
        image_file_name = op.join(image_root, noffset,
            xml_basename + ".JPEG")
        if not op.isfile(image_file_name):
            logging.info('not exist: {}'.format(image_file_name))
            continue
        im = cv2.imread(image_file_name, cv2.CV_LOAD_IMAGE_COLOR)
        gt, height, width = load_voc_xml(curr_file)
        draw_bb(im, [g['rect'] for g in gt], [g['class'] for g in gt])
        if out_folder != None:
            save_image(im, op.join(out_folder, xml_basename + '_bb.JPEG'))
        else:
            show_image(im)


def load_imagenet22k():
    file_name = '/home/jianfw/data/fall11_urls.txt'
    dest = '/home/jianfw/work/imagenet22k.cmap'
    if op.isfile(dest):
        u_names = load_list_file(dest)
        return u_names
    names = []
    with open(file_name, 'r') as fp:
        for i, line in enumerate(fp):
            parts = line.split(' ')
            if len(parts) > 0:
                parts = parts[0].split('_')
                if len(parts) > 0:
                    names.append(parts[0])
            if (i % 1000000) == 0:
                logging.info('{}-{}'.format(i, len(names)))
    u_names = list(set(names))
    logging.info(len(u_names))
    write_to_file('\n'.join(u_names), dest)
    return u_names


def expand_by_leaves(expand_list, imagenet22k_tree):
    mapped = []
    mapper = LabelToSynset()
    for e in expand_list:
        ss = mapper.convert(e)
        if type(ss) is not list:
            logging.info('Find {} in wordnet'.format(e))
            mapped.append(ss)
        else:
            good_ss = []
            for s in ss:
                nodes = imagenet22k_tree.search_nodes(noffset=synset_to_noffset(s))
                if len(nodes) > 0:
                    good_ss.append(s)
            if len(good_ss) == 1:
                logging.info('Find {} in wordnet'.format(e))
                mapped.append(good_ss[0])
            else:
                logging.info('Fail to find {} in wordnet'.format(e))

    for s in mapped:
        nodes = imagenet22k_tree.search_nodes(noffset=synset_to_noffset(s))
        if len(nodes) == 0:
            logging.info('Fail to find {} in the dataset tree'.format(s.name()))
            continue
        meta_list = []
        node = nodes[0]
        for leaf in node.iter_leaves():
            noffset = leaf.noffset
            meta_list.append({'name': str(get_nick_name(noffset_to_synset(noffset))),
                    'noffset': noffset,
                    'url': noffset_to_url(noffset)})
        write_to_file(yaml.dump(meta_list, default_flow_style=False),
                '/home/jianfw/work/taxonomy/expand_by_imagenet22k/{}.yaml'.format(get_nick_name(s)))


def get_imagenet22k_tree():
    cache_file = '/home/jianfw/work/taxonomy/dataset_tree/imagenet22k_tree.pkl'
    if op.isfile(cache_file):
        imagenet22k_tree = pkl.load(open(cache_file, 'r'))
        return imagenet22k_tree
    else:
        imagenet22k_noffset = load_imagenet22k()
        imagenet22k_synset = [noffset_to_synset(n) for n in imagenet22k_noffset]
        root_synset = wn.synset('entity.n.01')
        imagenet22k_tree = Tree(name=root_synset.name())
        imagenet22k_tree.add_features(noffset=synset_to_noffset(root_synset))
        update_with_synsets(imagenet22k_tree, imagenet22k_synset)
        pkl.dump(imagenet22k_tree, open(cache_file, 'w'))
        return imagenet22k_tree


def build_noffset_to_node(root):
    noffset_to_node = {}

    for node in root.iter_search_nodes():
        if hasattr(node, 'noffset'):
            noffset_to_node[node.noffset] = node
    return noffset_to_node


def populate_num_images_by_noffset(root, noffset_count):
    assert False, 'no longer used. use images_with_bb/images_no_bb'
    noffset_node = build_noffset_to_node(root)
    for noffset in noffset_count:
        if noffset in noffset_node:
            noffset_node[noffset].add_feature('images', 
                    noffset_count[noffset])

def count_num_annotation_images(annotation_root, image_root):
    noffset_count = {}
    for d in os.listdir(annotation_root):
        ensure_untar(d)
        if len(d) > 0 and d[0] == 'n':
            num = 0
            for f in glob.glob(op.join(annotation_root, d, '*.xml')):
                base_name = op.splitext(op.basename(f))[0]
                fn_im = op.join(image_root, d, base_name + '.JPEG')
                if op.isfile(fn_im):
                    im = cv2.imread(fn_im, cv2.IMREAD_COLOR)
                    if im is not None:
                        num = num + 1
            noffset_count[d] = num 
    return noffset_count

def count_num_images_3k(annotation_root, image_root):
    return count_num_annotation_images(annotation_root, image_root)

def build_imagenet3k():
    cache_file = '/home/jianfw/work/dataset_tree/imagenet3k_taxonomy_tree.pkl'
    if op.isfile(cache_file):
        with open(cache_file, 'r') as fp:
            root = pkl.load(fp)
            return root
    folder = '/home/jianfw/data/imagenetdet3k'
    all_file = glob.glob(op.join(folder, 'n*.tar.gz'))
    ss = []
    for file_name in all_file:
        base_name = op.basename(file_name)
        if len(base_name) < 7:
            continue
        noffset = base_name[:-7]
        ss.append(noffset_to_synset(noffset))
    root_synset = wn.synset('entity.n.01')
    root = Tree(name=root_synset.name())
    root.add_features(noffset=synset_to_noffset(root_synset))
    update_with_synsets(root, ss)
    write_to_file(pkl.dumps(root), cache_file)
    return root


def show_tree(root):
    ts = TreeStyle()
    ts.show_leaf_name = False
    def my_layout(node):
            F = TextFace(node.name, tight_text=True)
            add_face_to_node(F, node, column=0, position="branch-right")
    ts.layout_fn = my_layout
    root.show(tree_style=ts)

def keep_few(root, depth):
    assert depth != 0
    if depth == 1:
        all_children = []
        for c in root.children:
            all_children.append(c)
        for c in all_children:
            root.remove_child(c)
    else:
        for c in root.children:
            keep_few(c, depth - 1)

def test_noffset_imagenet():
    '''
    based on the file of ~/data/imagenet_detection, generate the noffsets.txt
    '''
    labelmap_f = './data/imagenet/labelmap.txt'
    noffset_f = './data/imagenet/noffsets.txt'
    labels = [s.strip() for s in read_to_buffer(labelmap_f).split('\n')]
    if labels[-1] == '':
        labels = labels[:-1]
    all_n_l = read_to_buffer('/home/jianfw/data/imagenet_detection').split('\n')
    label_noffset = {}
    for n_l in all_n_l:
        n_l = n_l.strip()
        if len(n_l) == 0:
            continue
        noffset = n_l[:9]
        label = n_l[9:].strip()
        label_noffset[label] = noffset
    noffsets = []
    for l in labels:
        if l in label_noffset:
            noffsets.append(label_noffset[l])
        elif l.replace('_', ' ') in label_noffset:
            noffsets.append(label_noffset[l.replace('_', ' ')])
        else:
            assert False
    write_to_file('\n'.join(noffsets), noffset_f)


def test_labels2noffsets():
    f = '/home/jianfw/code/darknet/data/coco.names'
    f = './data/voc20/labelmap.txt'
    labels = load_list_file(f)
    noffsets = labels2noffsets(labels)
    noffset_idx, _, _ = load_label_parent('./aux_data/yolo/9k.tree')
    maps = [noffset_idx[n] for n in noffsets]
    write_to_file('\n'.join(map(str, maps)), './output/voc20/map_to_yolo9k.txt')

def noffset_to_9000_idx():
    dataset = TSVDataset('imagenet')
    target_noffsets = read_to_buffer(dataset.get_noffsets_file()).split('\n')
    noffset_idx, noffset_parentidx, noffsets = load_label_parent('/home/jianfw/code/yolo-9000/darknet/data/9k.tree')
    target_idx = []
    for tnoffset in target_noffsets:
        target_idx.append(noffset_idx[tnoffset])
    write_to_file('\n'.join(map(str, target_idx)), './output/imagenet/map_to_yolo9000.txt')

def imagenet_label_to_noffset():
    dataset = TSVDataset('imagenet')
    label_names = dataset.load_labelmap()
    lines = read_to_buffer('./aux_data/imagenet200/ILSVRC2017.txt').split('\n')
    label_name_to_offset = {}
    for line in lines:
        if len(line.strip()) == 0:
            continue
        noffset = line[:9]
        assert noffset[0] == 'n'
        offset = int(noffset[1:])
        label_name_to_offset[line[9:].strip()] = noffset 
    assert len(label_name_to_offset) == 200
    result = []
    for label in label_names:
        if label in label_name_to_offset:
            noffset = label_name_to_offset[label]
        elif ' ' in label:
            label.replace(' ', '_')
            if label in label_name_to_offset:
                noffset = label_name_to_offset[label]
            else:
                assert False
        result.append(noffset)
    write_to_file('\n'.join(result), './data/imagenet/noffsets.txt')

def create_imagenet_tsv():
    label_tree_file = '/home/jianfw/code/yolo-9000/darknet/data/9k.tree'
    label_idx, _, noffsets = load_label_parent(label_tree_file)
    rows = imagenet_row_generator(label_idx, 500)
    tsv_writer(rows, '/raid/jianfw/work/imagenet22k_9ktree/train.tsv')
    labelmap = '\n'.join(label for label in noffsets)
    write_to_file(labelmap,
            '/raid/jianfw/work/imagenet22k_9ktree/labelmap.txt')
def imagenet_row_generator(label_idx, max_long_side):
    '''
    resize: if the smallest size is larger than
    '''
    image_root_folder = '/raid/jianfw/work/ImageNet22K_9K_images'
    image_files = []
    for noffset in label_idx:
        image_files += glob.glob(op.join(image_root_folder, noffset, '*'))
    logging.info('Total number of images: {}'.format(len(image_files)))
    np.random.seed(777)
    image_files = np.random.permutation(image_files)
    for i, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if im == None:
            logging.error('cannot open {}'.format(image_file))
            continue
        if im.shape[0] <= max_long_side and im.shape[1] <= max_long_side:
            with open(image_file, 'rb') as fp:
                imdata = fp.read()
            debug = ','.join(map(str, im.shape))
        else:
            m = max(im.shape[0], im.shape[1])
            ratio = 1.0 * max_long_side / m
            size = map(lambda x: int(np.round(x*ratio)),
                im.shape[:2])
            size = (size[1], size[0]) 
            im2 = cv2.resize(im, size)
            imdata = cv2.imencode('.jpg', im2)[1]
            debug = '{}->{}'.format(','.join(map(str, im.shape)), 
                    ','.join(map(str, im2.shape)))
        sig = op.splitext(op.basename(image_file))[0]
        noffset = op.basename(op.dirname(image_file))
        l_idx = str(label_idx[noffset])
        if (i % 500) == 0:
            logging.info('{}/{}: {}'.format(i, len(image_files), debug))
        yield sig, l_idx, base64.b64encode(imdata)


def untar_imagenet22():
    _, label_parentidx = load_label_parent('/home/jianfw/code/yolo-9000/darknet/data/9k.tree')
    logging.info(len(label_parentidx))
    tar_folder = '/vighd/data/ImageNet22K'
    target_root_folder = '/raid/jianfw/work/ImageNet22K_9K_images'
    for noffset in label_parentidx:
        tar_file = op.join(tar_folder, noffset + '.tar')
        if not os.path.isfile(tar_file):
            logging.info('not exists: {}'.format(tar_file))
            continue
        logging.info('extract: {}'.format(tar_file))
        t = tarfile.open(tar_file)
        target_folder = op.join(target_root_folder, noffset)
        t.extractall(target_folder)

def untar_imagenet3k():
    target_root_folder = '/home/jianfw/work/imagenet3k/annotation'
    for tar_file in glob.glob('/home/jianfw/data/imagenetdet3k/n*.tar.gz'):
        if not os.path.isfile(tar_file):
            logging.info('not exists: {}'.format(tar_file))
            continue
        noffset = op.basename(tar_file)[: -7]
        logging.info('extract: {} to {}'.format(tar_file, noffset))
        t = tarfile.open(tar_file)
        t.extractall(target_root_folder)

def test_merge_labelset():
    root_synset = wn.synset('physical_entity.n.01')
    root = Tree(name=root_synset.name())
    root.add_features(synset=root_synset)
    #labels = ['Norfolk terrier', 'Yorkshire terrier', 'terrier']
    labels = ['n02094114', 'n02094433']
    with open('/home/jianfw/code/yolo-9000/darknet/data/9k.labels', 'r') as fp:
    #with open('/home/jianfw/code/yolo-9000/darknet/data/imagenet.labels.list', 'r') as fp:
        lines = fp.readlines()
    labels = [line.split(' ')[0] for line in lines]
    logging.info(len(labels))
    ss = []
    for i, label in enumerate(labels):
        assert len(label) > 1
        pos, offset = label[0], int(label[1:])
        ss.append(wn.synset_from_pos_and_offset(pos, offset))
        #if i >= 899:
            #break
    logging.info('down loading')
    update_with_synsets(root, ss)
    logging.info(tree_size(root))
    root = prune(root)
    check_prune(root)
    logging.info(tree_size(root))
    write_to_file(root.get_ascii(attributes=['nick_name']), 'tmp.txt')

def compare_tree(label_parentidx_file2):
    f1 = '/home/jianfw/code/yolo-9000/darknet/data/9k.tree'
    def convert(c):
        lines = c.split('\n')
        parts = []
        for line in lines:
            parts.append([p.strip() for p in line.split(' ')])
        r = {}
        for p in parts:
            if len(p) == 1 and p[0] == '':
                continue
            r[p[0]] = parts[int(p[1])][0] if int(p[1]) != -1 else ''
        return r

    c1 = read_to_buffer(f1)
    r1 = convert(c1)
    c2 = read_to_buffer(label_parentidx_file2)
    r2 = convert(c2)
    for k1 in r1:
        assert r1[k1] == r2[k1]

def test_gen_term_list():
    tax_folder = '/home/jianfw/data/10KDetector/'
    term_list = '/home/jianfw/work/abc.txt'
    gen_term_list(tax_folder, term_list)


def test_gen_noffset():
    tax_input_folder = '/home/jianfw/work/10KDetector/'
    tax_output_folder = '/home/jianfw/work/10KDetector2/'
    gen_noffset(tax_input_folder, tax_output_folder)


def test_gen_cls_specific_th():
    tree_file = './data/office100_v1/tree.txt'    
    th_file = './work/abc.txt'
    gen_cls_specific_th(tree_file, th_file)

def test_build_tree_from_tree_file():
    root = build_tree_from_tree_file()
    #write_to_file(root.get_ascii(),
            #'/home/jianfw/work/yolo9000_imagenet/ete2_tree.txt')
    keep_few(root, 4)
    write_to_file(root.get_ascii(attributes=['nick_name']), 'tmp.txt')

    root.show()

def build_tree_from_tree_file():
    file_name = '/home/jianfw/code/yolo-9000/darknet/data/9k.tree'
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
    label_parient = {}
    noffsets = []
    label_node = {}
    root = Tree()
    root_synset = wn.synset('physical_entity.n.01')
    root.name = root_synset.name() 
    root.add_feature('synset', root_synset)
    for line in lines:
        label, parientid = (p.strip() for p in line.split(' '))
        parientid = int(parientid)
        noffsets.append(label)
        label_parient[label] = parientid
        s = wn.synset_from_pos_and_offset(label[0], int(label[1:]))
        if parientid == -1:
            c = root.add_child(name=label)
        else:
            parientnode = label_node[noffsets[parientid]]
            c = parientnode.add_child(name=label)
        c.add_features(synset=s, nick_name=get_nick_name(s))
        label_node[label] = c

    s = breadth_first_print_tree(root)
    write_to_file('\n'.join(s), 
            '/home/jianfw/work/yolo9000_imagenet/parient_child_tree.txt')
    s = child_parent_print_tree(root)
    label_parentidx_file2 = '/home/jianfw/work/tmp/b.txt'
    write_to_file('\n'.join(s), label_parentidx_file2)
    compare_tree(label_parentidx_file2)
    check_prune(root)

    prune_root(root)

    return root

def untar_all_imagenet22k():
    tar_folder = '/vighd/data/ImageNet22K'
    all_folder = os.listdir(tar_folder)
    for folder in all_folder:
        if len(folder) < 8:
            logging.info('unkown: {}'.format(folder))
            continue
        noffset = op.splitext(folder)[0]
        logging.info('untar: {}'.format(noffset))
        ensure_untar(noffset)

def create_imagenet3k_tsv():
    home_root = '/raid/jianfw'
    data_sources = []
    image_root = op.join(home_root, 'work', 'ImageNet22K')
    annotation_root = op.join(home_root, 'work', 'imagenet3k', 'annotation',
            'Annotation')
    labels = []
    def gen_rows():
        for noffset in os.listdir(annotation_root):
            ensure_untar(noffset)
            labels.append(noffset)
            logging.info(noffset)
            if noffset is None:
                continue
            rows = gen_image_gt_by_noffset(noffset, annotation_root,
                    image_root)
            for basename, image_file_name, gt, height, width in rows:
                with open(image_file_name, 'r') as fp:
                    im_raw = fp.read()
                im = cv2.imdecode(np.fromstring(im_raw, np.uint8), 
                        cv2.IMREAD_COLOR)
                if im is None or im.shape[0] == 0:
                    logging.info('image not available: {}'.format(
                        image_file_name))
                    continue
                image_changed = False
                if min(im.shape[:2]) > 448:
                    logging.info('image size is too large. Resizing: {}'.format(basename))
                    ratio = 448. / min(im.shape[: 2])
                    dsize = (int(im.shape[1] * ratio), int(im.shape[0] *
                        ratio))
                    im = cv2.resize(im, dsize)
                    image_changed = True

                if height != im.shape[0] or width != im.shape[1]:
                    hratio = im.shape[0] / float(height)
                    wratio = im.shape[1] / float(width)
                    for g in gt:
                        g['rect'][0] = g['rect'][0] * wratio
                        g['rect'][2] = g['rect'][2] * wratio
                        g['rect'][1] = g['rect'][1] * wratio
                        g['rect'][3] = g['rect'][3] * wratio
                if not image_changed:
                    im_str = base64.b64encode(im_raw)
                else:
                    im_str = encoded_from_img(im)
                yield (basename, json.dumps(gt), im_str)
    dataset = TSVDataset('imagenet3k_448')
    tsv_writer(gen_rows(), dataset.get_trainval_tsv())
    write_to_file('\n'.join(labels), dataset.get_labelmap_file())


def create_imagenet22k_tsv():
    home_root = '/raid/jianfw'
    image_root = op.join(home_root, 'work', 'ImageNet22K')
    image_root = '/vighd/data/ImageNet22K' 
    labelmap = []
    def gen_rows():
        for i, noffset in enumerate(os.listdir(image_root)):
            logging.info('{}-{}'.format(i, noffset))
            images = glob.glob(op.join(image_root, noffset, '*.JPEG'))
            #for image_file_name in images:
                #im = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
                #if im is None or im.size == 0:
                    #logging.info('invalid {}'.format(image_file_name))
                    #continue
                #with open(image_file_name, 'r') as fp:
                    #im_str = base64.b64encode(fp.read())
                #gt = [{'rect': [0, 0, 0, 0], 'class': noffset}]
                #image_key = op.splitext(op.basename(image_file_name))[0]
                #yield image_key, json.dumps(gt), im_str
            labelmap.append(noffset)
    dataset = TSVDataset('imagenet22k_448')
    gen_rows()
    #tsv_writer(gen_rows(), dataset.get_trainval_tsv())
    write_to_file('\n'.join(labelmap), dataset.get_labelmap_file())
        

def resize_dataset():
    data = 'imagenet22k'
    dataset = TSVDataset(data)
    rows = tsv_reader(dataset.get_trainval_tsv())
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info(i)
            im = img_from_base64(row[-1])
            if im is None or im.size == 0:
                logging.info('the image is empty: {}'.format(im[0]))
            elif min(im.shape[0:2]) > 448:
                ratio = 448. / min(im.shape[:2])
                dsize = (int(im.shape[1] * ratio), int(im.shape[0] * ratio))
                rects = json.loads(row[1])
                for rect in rects:
                    if not all(r == 0 for r in rect['rect']):
                        rect['rect'] = map(lambda x: x * ratio, rect['rect'])
                im2 = cv2.resize(im, dsize)
                yield row[0], json.dumps(rects), encoded_from_img(im2)
            else:
                yield row
    ndataset = TSVDataset(data + '_448')
    tsv_writer(gen_rows(), ndataset.get_train_tsv())


def test_central_overlap():
    data = 'voc20'
    dataset = TSVDataset(data)
    rows = tsv_reader(dataset.get_train_tsv())
    total = 0
    total_bb = 0
    overlapped = 0
    having_box = 0
    #anchor = [(8, 8)]
    #anchor = [(4, 8), (8, 4)]
    anchor = [(4, 4), (4, 8), (8, 4), (8, 8)]
    anchor = [(1.08,
    1.19),
    (3.42,
    4.41),
    (6.63,
    11.38),
    (9.42,
    5.11),
    (16.62,
    10.52)]
    for row in rows:
        rects = json.loads(row[1])
        total = total + 1
        S = set()
        plotted = False
        for i, rect in enumerate(rects):
            if i == 0:
                having_box = having_box + 1
            total_bb = total_bb + 1
            coords = rect['rect']
            center_x = (coords[0] + coords[2]) / 2
            center_y = (coords[1] + coords[3]) / 2
            center_x = int((center_x + 31) / 32)
            center_y = int((center_y + 31) / 32)
            w = (coords[2] - coords[0] + 31) / 32.0
            h = (coords[3] - coords[1] + 31) / 32.0
            idx_anchor = np.argmax([iou(a, (w, h)) for a in anchor])
            v = (center_x, center_y, idx_anchor)
            if v in S:
                overlapped = overlapped + 1
                if not plotted:
                    im = img_from_base64(row[2])
                    all_rect = [r['rect'] for r in rects]
                    all_class = [r['class'] for r in rects]
                    all_class[i] = 's_' + all_class[i]
                    draw_bb(im, all_rect, all_class)
                    save_image(im,
                            '/home/jianfw/work/tmp/{}.jpg'.format(row[0]))
            else:
                S.add(v)

    logging.info('{}-{}-{}'.format(total, having_box, overlapped))
    logging.info('{}-{}'.format(total_bb, overlapped))


def test_visualize():
    tsv_image = \
        '/raid/data/crawl_office_v2/train.tsv'
    out_folder = '/raid/jianfw/work/crawl_office_v2/images'
    #visualize_tsv(tsv_image, tsv_image, out_folder=out_folder, label_idx=0)
    #visualize_tsv(tsv_image, tsv_image, out_folder=out_folder)
    #visualize_tsv2('crawl_office_v2', 'Indoor')
    #visualize_tsv2('crawl_office_v2', 'utensil')
    #visualize_tsv2('imagenet3k_448', 'trainval', 'n02992529')
    #visualize_tsv2('imagenet3k_448', 'trainval', 'n03216710')
    data = 'office_v2.12_with_bb'
    populate_dataset_details(data)
    visualize_tsv2(data, 'train', 'paper cup')
    #visualize_tsv2(data, 'test', 'paper cup')

def add_images(source, ext_data, ext_label, ext_label_to_target, out_data):
    dataset = default_data_path(data)
    out_dataset = default_data_path(out_data)


    rows = two_file_reader(dataset['source'], ext_dataset['source'], ext_label,
            ext_label_to_target)

    tsv_writer(rows, out_data['source'])

    os.symlink(dataset['test_source'], out_data['test_source'])
    os.symlink(dataset['test_source_idx'], out_data['test_source_idx'])
    os.symlink(dataset['labelmap'], out_data['labelmap'])


def test_merge_image():
    source = '/home/jianfw/code/quickdetection/data/voc20/train.tsv'
    source_label = \
    '/home/jianfw/code/quickdetection/output/data/voc20/train_label_remove_bottle_bb_0.1.tsv'
    add_image_rule = 'coco:bottle:500'
    out_source = '/home/jianfw/work/tmp.tsv'
    merge_image(source, source_label, add_image_rule, out_source)

def merge_image(source, source_label, source_labelmap, add_image_rule, out_source):
    '''
    source: tsv to extract the image
    source_label: label tsv
    add_image_rule: dataset:label
    out_source: one tsv file name 
    '''
    def two_file_reader(source, source_label, ext_source, ext_label,
            max_num, ext_label_to_target, is_bbox=False):
        source_rows = tsv_reader(source) if source else None
        source_label_rows = tsv_reader(source_label) if source_label else None
        if source_rows and source_label_rows:
            for row, label_row in zip(source_rows, source_label_rows):
                assert row[0] == label_row[0]
                out_row = [row[0], label_row[1], row[2]]
                yield out_row
        elif source_rows and source_label_rows == None:
            for row in source_rows:
                yield row
        elif source_rows == None and source_label_rows:
            assert False

        ext_source_rows = tsv_reader(ext_source)
        all_target_label = [target_label for target_label in set(ext_label_to_target.values()) if
                target_label]
        required = {}
        if ext_label == 'anyuseful':
            for tl in all_target_label:
                required[tl] = max_num
        for row in ext_source_rows:
            labels = json.loads(row[1])
            to_be_removed = []
            if max_num <= 0:
                break
            if ext_label == 'anyuseful' or any((label['class'] == ext_label for label in labels)):
                labels2 = []
                for label in labels:
                    if not is_bbox:
                        label['rect'] = [0, 0, 0, 0]
                    if ext_label_to_target[label['class']] != None:
                        label['class'] = ext_label_to_target[label['class']]
                        labels2.append(label)
                if len(labels2) > 0:
                    got_quota = False
                    for label in labels2:
                        if required[label['class']] > 0:
                            required[label['class']] = required[label['class']]-1
                            got_quota = True
                            break
                    if got_quota:
                        row[1] = json.dumps(labels2)
                        yield row

    ext_data, target_label, withbbinfo, max_num = add_image_rule.split(':')
    assert withbbinfo == 'withbb' or withbbinfo == 'nobb'
    max_num = int(max_num)
    ext_dataset = default_data_path(ext_data)
    from taxonomy import create_labelmap_map
    labelmap_map = create_labelmap_map(ext_dataset['labelmap'], source_labelmap)
    is_bbox = True if withbbinfo == 'withbb' else False
    if target_label == 'anyuseful': 
        ext_label = 'anyuseful'
    else:
        ext_label = create_labelmap_map(source_labelmap,
                ext_dataset['labelmap'])[target_label]
    assert ext_label
    rows = two_file_reader(source, source_label, ext_dataset['source'], 
            ext_label, max_num, labelmap_map, is_bbox)
    tsv_writer(rows, out_source)

def test_update_yolo_test_proto():
    update_yolo_test_proto('./output/imagenet_darknet19_A_noreorg_noextraconv/test.prototxt', 
            'coco',
            './output/coco/yolo_map_9k.txt',
            './output/imagenet_darknet19_A_noreorg_noextraconv/test.coco.prototxt')



class BatchProcess(object):
    def __init__(self, all_resource, all_task, processor):
        self._all_task = Queue()
        for task in all_task:
            self._all_task.put(task)

        self._all_resouce = Queue()
        for resource in all_resource:
            self._all_resouce.put(resource)

        self._task_description = '{}\n\n{}'.format(pformat(all_task),
                pformat(all_resource))

        self._availability_check = True
        self._processor = processor
        self._in_use_resource_file = op.expanduser('~/.in_use_resource.yaml')

    def _run_in_lock(self, func):
        #import portalocker
        #logging.info('begin to lock')
        #with portalocker.Lock('/tmp/process_tsv.lock') as fp:
            #portalocker.lock(fp, portalocker.LOCK_EX)
            #logging.info('end to lock')
        result = func()
        return result

    def _try_use_resource(self, resource):
        def use_resource(resource):
            in_use = self._is_in_use(resource)
            if not in_use:
                in_use_resource = self._load_save_valid_in_use_status()
                in_use_resource.append({'pid': os.getpid(),
                    'ip': resource[0].get('ip', 'localhost'),
                    'port': resource[0].get('-p', 22),
                    'gpus': resource[1]})
                logging.info('writting to {}'.format(self._in_use_resource_file))
                write_to_yaml_file(in_use_resource, self._in_use_resource_file)
            else:
                logging.info('resource is in use: {}'.format(pformat(resource)))
            return not in_use

        return self._run_in_lock(lambda: use_resource(resource))

    def _release_resouce(self, resource):
        def release_resource(resource):
            in_use_resource = self._load_save_valid_in_use_status()
            my_pid = os.getpid()
            to_be_removed = []
            for record in in_use_resource:
                if record['pid'] == my_pid and \
                        record.get('ip', '0') == resource[0].get('ip', '0') and \
                        record.get('port', '0') == resource[0].get('-p', 22) and \
                        all(g1 == g2 for g1, g2 in izip(record['gpus'],
                            resource[1])):
                    to_be_removed.append(record)

            logging.info('to be removed: {}'.format(pformat(to_be_removed))) 

            for r in to_be_removed:
                in_use_resource.remove(r)
            write_to_yaml_file(in_use_resource, self._in_use_resource_file)

        self._run_in_lock(lambda: release_resource(resource))

    def _load_save_valid_in_use_status(self):
        if not op.isfile(self._in_use_resource_file):
            in_use_resources = []
        else:
            in_use_resources = load_from_yaml_file(self._in_use_resource_file)
        in_use_resources2 = []
        for record in in_use_resources:
            in_use_process_id = record['pid']
            in_use_ip = record.get('ip', '0')
            in_use_gpus = record['gpus']
            from remote_run import process_exists
            if not process_exists(in_use_process_id):
                logging.info('{} is not running. ignore that record'.format(
                    in_use_process_id))
                continue
            in_use_resources2.append(record)
        write_to_yaml_file(in_use_resources2,
                self._in_use_resource_file)
        return in_use_resources2

    def _check_if_in_use(self, resource):
        result = False
        if op.isfile(self._in_use_resource_file):
            in_use_resources = load_from_yaml_file(self._in_use_resource_file)
            in_use_resources2 = []
            for record in in_use_resources:
                in_use_process_id = record['pid']
                in_use_ip = record.get('ip', '0')
                in_use_gpus = record['gpus']
                from remote_run import process_exists
                if not process_exists(in_use_process_id):
                    logging.info('{} is not running. ignore that record'.format(
                        in_use_process_id))
                    continue
                in_use_resources2.append(record)
                if resource[0].get('ip', '0') == in_use_ip and \
                        record.get('port', '0') == resource[0].get('-p', 22) and \
                        any(g in resource[1] for g in in_use_gpus):
                    result = True
                    logging.info('resouce is in use \n{}'.format(pformat(resource)))
                    break
            write_to_yaml_file(in_use_resources2,
                    self._in_use_resource_file)
        else:
            logging.info('{} not exists'.format(self._in_use_resource_file))
        return result

    def _is_in_use(self, resource):
        return self._run_in_lock(lambda:  self._check_if_in_use(resource))

    def run(self):
        self._in_progress = []
        self._has_synced = {}
        while True:
            in_progress = []
            for resource, task, p in self._in_progress:
                if not p.is_alive():
                     p.join()
                     self._release_resouce(resource)
                     if p.exitcode != 0:
                        for _, __, x in self._in_progress:
                            x.terminate()
                        logging.info(pformat(resource))
                        logging.info(pformat(task))
                        assert False
                     
                     self._all_resouce.put(resource)
                else:
                    in_progress.append((resource, task, p))
            self._in_progress = in_progress

            if self._all_task.empty():
                break
            if self._all_resouce.empty():
                time.sleep(5)
                continue
            resource = self._all_resouce.get()

            if self._availability_check:
                avail = True
                if not gpu_available([resource]):
                    logging.info('{} is occupied from nvidia-smi'.format(
                        resource))
                    avail = False
                if avail and self._is_in_use(resource):
                    logging.info('{} is in possesion of other process'.format(
                        resource))
                    avail = False
                if not avail:
                    self._all_resouce.put(resource)
                    logging.info('resouce ({}) is not available. #task left {}'.format(
                        resource,
                        self._all_task.qsize()))
                    time.sleep(5)
                    continue

            if not self._try_use_resource(resource):
                logging.info('fails to try to use {}'.format(pformat(resource)))
                continue

            logging.info('resource ({}) is available'.format(resource))

            task = self._all_task.get()
            if len(resource[0]) > 0 and resource[0]['ip'] not in self._has_synced:
                sync_qd(ssh_info=resource[0], delete=True)
                self._has_synced[resource[0]['ip']] = True

            p = mp.Process(target=self._processor, args=(resource, task))
            p.start()
            self._in_progress.append((resource, task, p))
            if is_cluster(resource[0]):
                time.sleep(5)

        for resource, task, p in self._in_progress:
            p.join()
            self._release_resouce(resource)

        #notify_by_email('Job finished', self._task_description)


def remove_bb(source_tsv, dst_tsv, **kwargs):
    stat = {}

    random.seed(2)
    t = TSVTransformer()
    t.Process(source_tsv, dst_tsv, lambda row: remove_bb_processor(row, stat,
        **kwargs))

    pprint(stat)

def remove_bb_processor(row, stat = {}, **kwargs):
    '''
    input: row is a tuple or a list. Each entry is a column entry in the tsv
    file
    output: a list of entry, which might be saved into a new tsv file
    '''
    assert len(row) == 3, len(row)

    label_idx = 1

    stat['total_line'] = stat.get('total_line', 0) + 1
    label_col = row[label_idx]
    labels = json.loads(label_col)

    bb_removal_label = kwargs['remove_bb_label']
    remove_bb_label_prob = kwargs.get('remove_bb_label_prob', 1)

    all_label = bb_removal_label.split(',')

    remove = False
    for label in labels:
        if label['class'] in all_label or bb_removal_label == 'all':
            if random.random() <= remove_bb_label_prob:
                remove = True
                break
    if remove:
        stat['remove'] = stat.get('remove', 0) + 1
        for label in labels:
            label['rect'] = [0, 0, 0, 0]
            
    label_col = json.dumps(labels)

    return [row[0], label_col]

def gen_coco_noffset_map():
    darknet_coco_labels = load_list_file('/home/jianfw/code/darknet/data/coco.names')
    darknet_coco_idx = load_list_file('/home/jianfw/code/darknet/data/coco9k.map')
    darknet_9k_noffsets = load_list_file('/home/jianfw/code/darknet/data/9k.labels')
    darknet_coco_idx = [int(n) for n in darknet_coco_idx]
    darknet_coco_noffset = [darknet_9k_noffsets[i] for i in darknet_coco_idx]
    label_to_noffset = {l:n for l, n in izip(darknet_coco_labels,
            darknet_coco_noffset)}
    d = []
    for l in label_to_noffset:
        d.append({'name': l, 'noffset': label_to_noffset[l]})

    write_to_yaml_file(d, './aux_data/label_to_noffset/for_coco.yaml')

def split_labels():
    tsv = TSVDataset('cifar10')

    tsv_first = TSVDataset('cifar10_first5')
    tsv_second = TSVDataset('cifar10_second5')

    th = 5

    def gen_first(t):
        if t == 'train':
            rows = tsv_reader(tsv.get_train_tsv())
        else:
            rows = tsv_reader(tsv.get_test_tsv_file())
        for row in rows:
            if int(row[1]) < th:
                yield row

    def gen_second(t):
        if t == 'train':
            rows = tsv_reader(tsv.get_train_tsv())
        else:
            rows = tsv_reader(tsv.get_test_tsv_file())
        for row in rows:
            if int(row[1]) >= th:
                row[1] = str(int(row[1]) - th)
                yield row
    tsv_first = TSVDataset('cifar10_first{}'.format(th))
    tsv_second = TSVDataset('cifar10_second{}'.format(th))
    labels = tsv.load_labelmap()
    tsv_writer(gen_first('train'), tsv_first.get_train_tsv())
    write_to_file('\n'.join(labels[:th]), tsv_first.get_labelmap_file())
    tsv_writer(gen_first('test'), tsv_first.get_test_tsv_file())

    tsv_writer(gen_second('train'), tsv_second.get_train_tsv())
    tsv_writer(gen_second('test'), tsv_second.get_test_tsv_file())
    write_to_file('\n'.join(labels[th:]), tsv_second.get_labelmap_file())


def update_rects_within_image(rects, im=None):
    to_removed = []
    for rect in rects:
        x1, y1, x2, y2 = rect['rect']
        x1 = max(0, x1)
        if im is not None:
            x1 = min(im.shape[1], x1)
        x2 = max(0, x2)
        if im is not None:
            x2 = min(im.shape[1], x2)
        y1 = max(0, y1)
        if im is not None:
            y2 = min(im.shape[0], y2)
        rect['rect'] = [x1, y1, x2, y2]

    for rect in rects:
        x1, y1, x2, y2 = rect['rect']
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            to_removed.append(rect)

    for rect in to_removed:
        rects.remove(rect)

    return rects

def create_toy_dataset():
    data = 'voc20'
    out_data = 'toy'
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)

    for split in ['train', 'test']:
        rows = tsv_reader(dataset.get_data(split))
        def gen_rows():
            for i, row in enumerate(rows):
                if i > 10:
                    break
                yield row
        tsv_writer(gen_rows(), out_dataset.get_data(split))
    populate_dataset_details(out_data)


def merge_labels():
    def srclabel_to_targetlabel(label):
        return 'drink'
        #if 'ColaPot' in label:
            #return 'ColaPot'
        #elif 'ColaCan' in label:
            #return 'ColaCan'
        #else:
            #return label
    src_data = 'CocoBottle1024'
    src_dataset = TSVDataset(src_data)
    dest_data = '{}Drink'.format(src_data)
    dest_dataset = TSVDataset(dest_data)
    splits = ['train', 'test']
    labelmap = src_dataset.load_labelmap()
    mapper = {l: srclabel_to_targetlabel(l) for l in labelmap}
    for split in splits:
        f = src_dataset.get_data(split)
        t = dest_dataset.get_data(split)
        rows = tsv_reader(f)
        def gen_rows():
            for row in rows:
                rects = json.loads(row[1])
                for rect in rects:
                    rect['class'] = mapper[rect['class']]
                row[1] = json.dumps(rects)
                yield row
        tsv_writer(gen_rows(), t)
    populate_dataset_details(dest_data)

def create_cocobottle():
    root_folder = '/home/jianfw/data/raw_data/CocoBottle'
    all_image_file = glob.glob(op.join(root_folder, '*'))
    all_image_file = [image_file for image_file in all_image_file if \
            image_file.endswith('JPG') or \
            image_file.endswith('jpg')]
    all_info = []
    for image_file in all_image_file:
        anno_file = '{}.txt'.format(op.splitext(image_file)[0])
        anno_lines = load_list_file(anno_file)
        rects = []
        for line in anno_lines:
            splits = line.split(',')
            if len(splits) != 5:
                logging.info('skip: {}'.format(line))
                continue
            x0, y0, width, height = [float(s) for s in splits[1:]]
            label = splits[0].decode('ascii', errors='ignore').encode()
            assert type(label) == str
            rect = {'class': label, 'rect': [x0,
                y0,
                x0 + width,
                y0 + height]}
            rects.append(rect)
        if len(rects) < 5:
            continue
        all_info.append((image_file, rects))

        #im = cv2.imread(image_file, cv2.IMREAD_COLOR)
        #draw_bb(im, [r['rect'] for r in rects], 
                #[r['class'] for r in rects])
        #show_image(im)
    logging.info('total = {}'.format(len(all_info)))
    random.seed(777)
    random.shuffle(all_info)
    test_info = all_info[:25]
    train_info = all_info[25:]
    def gen_rows(curr_info):
        for image_file, rects in curr_info:
            key = op.basename(image_file)
            im = cv2.imread(image_file, cv2.IMREAD_COLOR)
            assert type(key) == str
            yield key, json.dumps(rects), base64.b64encode(read_to_buffer(image_file))
    data = 'CocoBottle'
    shutil.rmtree('data/{}'.format(data))
    dataset = TSVDataset(data)
    tsv_writer(gen_rows(train_info), dataset.get_data('train'))
    tsv_writer(gen_rows(test_info), dataset.get_data('test'))
    populate_dataset_details(data)

def create_logs18():
    raw_folder = '/home/jianfw/data/raw_data/logs18'
    all_im_file = glob.glob(op.join(raw_folder, '*'))
    random.seed(777)
    random.shuffle(all_im_file)
    test_image = all_im_file[:5]
    train_image = all_im_file[5:]
    pass

def resize_tsv():
    data = 'WIDER_FACE'
    data = 'CocoBottle'
    longest_side = 1024
    smallest_side = 512
    out_data = '{}{}'.format(data, longest_side)
    source_set = TSVDataset(data)
    dest_set = TSVDataset(out_data)
    splits = ['train', 'test']
    for split in splits:
        if not op.isfile(source_set.get_data(split)):
            continue
        rows = tsv_reader(source_set.get_data(split))
        def gen_rows():
            for row in rows:
                im = img_from_base64(row[-1])
                max_side = max(im.shape[:2])
                min_side = min(im.shape[:2])
                if longest_side > 0 and max_side > longest_side:
                    ratio = 1. * longest_side / max_side
                    out_im = cv2.resize(im, (int(im.shape[1] * ratio), 
                        int(im.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
                    rects = json.loads(row[1])
                    for rect in rects:
                        for i, r in enumerate(rect['rect']):
                            rect['rect'][i] = r * ratio
                    yield row[0], json.dumps(rects), encoded_from_img(out_im)
        tsv_writer(gen_rows(), dest_set.get_data(split))
    populate_dataset_details(out_data)

def check_target_overlap():
    data = 'WIDER_FACE'
    split = 'train'
    net_input = 416 * 5

    dataset = TSVDataset(data)
    tsv_file = dataset.get_data(split)
    rows = tsv_reader(tsv_file)
    
    total = 0
    good = 0
    for i, row in enumerate(rows):
        pos_to_count = {}
        im = img_from_base64(row[-1])
        assert im is not None
        rects = json.loads(row[1])
        for rect in rects:
            r = rect['rect']
            x_cent = (r[0] + r[2]) / 2. / im.shape[1]
            y_cent = (r[1] + r[3]) / 2. / im.shape[0]
            qx = int(np.floor(x_cent * net_input / 32))
            qy = int(np.floor(y_cent * net_input / 32))
            key = (qx, qy)
            if key not in pos_to_count:
                pos_to_count[key] = 0
            pos_to_count[key] = pos_to_count[key] + 1
        counts = [pos_to_count[key] for key in pos_to_count]
        counts.sort(reverse=True)
        if (i % 100) == 0:
            logging.info('{}-{}'.format(i, pformat(counts[:5])))
        total = total + len(rects)
        good = good + len([c for c in counts if c == 1])
    logging.info('{}-{}-{}'.format(good, total, 1. * good / total))

def check_wider_face():
    data = 'WIDER_FACE'
    dataset = TSVDataset(data)
    splits = ['train', 'test']
    input_size = 416
    for split in splits:
        fname = dataset.get_data(split)
        rows = tsv_reader(fname)
        for key, str_rects, str_im in rows:
            rects = json.loads(str_rects)
            im = img_from_base64(str_im)
            im.shape[0]
        pass
    pass

def create_wider_face():
    raw_data_root = '/home/jianfw/data/raw_data/WIDER_FACE'
    def wider_face_load_annotation(txt_file, image_folder, shuffle=False): 
        all_line = load_list_file(txt_file)
        i = 0
        all_info = []
        while i < len(all_line):
            file_name = all_line[i]
            rects = []
            num_bb = int(float(all_line[i + 1]))
            for j in xrange(num_bb):
                line = all_line[i + 2 + j]
                info = [float(s.strip()) for s in line.split(' ')]
                assert len(info) == 10
                x1, y1, w, h = info[:4]
                rect = {'rect': [x1, y1, x1 + w, y1 + h], 'class': 'face'}
                rects.append(rect)
            all_info.append((file_name, rects))
            i = i + 2 + num_bb
        if shuffle:
            random.shuffle(all_info)
        for i, (file_name, rects) in enumerate(all_info):
            if (i % 100) == 0:
                logging.info('{}/{}'.format(i, len(all_info)))
            full_file_name = op.join(image_folder, file_name)
            #im = cv2.imread(full_file_name, cv2.IMREAD_COLOR)
            #draw_bb(im, [r['rect'] for r in rects], [r['class'] for r in rects])
            #show_image(im)
            yield file_name, json.dumps(rects), base64.b64encode(read_to_buffer(full_file_name))
    splits_in_tsv = ['train', 'test']
    splits_in_origin = ['train', 'val']
    name = 'WIDER_FACE'
    for split_in_tsv, split_in_origin in izip(splits_in_tsv, splits_in_origin):
        txt_file = op.join(raw_data_root, 'wider_face_split',
            'wider_face_{}_bbx_gt.txt'.format(split_in_origin))
        folder = op.join(raw_data_root, 'WIDER_{}'.format(split_in_origin), 'images')
        tsv_writer(wider_face_load_annotation(txt_file, folder, True), 
                op.join('data', 'WIDER_FACE', '{}.tsv'.format(split_in_tsv)))
    populate_dataset_details('WIDER_FACE')

def upload_carpk_to_philly():
    philly_upload_dir('data/CARPK', 'data/qd_data', vc='input')
    for select_image in [5, 10, 50]:
        for select_bb in [5, 10, 25, 50]:
            if select_bb == 25 and select_image != 5:
                continue
            init_data = 'CARPK_select.{}.{}.nokeep'.format(
                    select_image,
                    select_bb)
            philly_upload_dir('data/{}'.format(init_data),
                    'data/qd_data', vc='input')

def distribute_dataset(name):
    philly_upload_dir('data/{}'.format(name), 'data/qd_data', vc='input')

    # copy the data to vig-gpu02/raid
    source_folder = 'data/{}'.format(name)
    shutil.copytree(source_folder, op.join('gpu02_raid', 'data', name))

    # move the data to glusterfs
    shutil.move(source_folder, '/home/jianfw/glusterfs/public/data/')


def get_all_tree_data():
    import qd_common
    return qd_common.get_all_tree_data()
    #names = os.listdir('./data')
    #return [name for name in names 
        #if op.isfile(op.join('data', name, 'root.yaml'))]

def gen_html_tree_view(data):
    import tsv_io
    return tsv_io.gen_html_tree_view(data)
    #dataset = TSVDataset(data)
    #file_name = op.join(dataset._data_root, 'root.yaml')
    #with open(file_name, 'r') as fp:
        #config_tax = yaml.load(fp)
    #tax = Taxonomy(config_tax)
    #def gen_html_tree_view_rec(root):
        #'''
        #include itself
        #'''
        #if len(root.children) == 0:
            #s = u"<li data-jstree='{{\"icon\":\"glyphicon glyphicon-leaf\"}}'>{}</li>".format(root.name)
            #return s
        #else:
            #result = []
            #result.append('<li><span>{}</span>'.format(root.name))
            #result.append('<ul>')
            #for c in root.children:
                #r = gen_html_tree_view_rec(c)
                #result.append(r)
            #result.append('</ul>')
            #result.append('</li>')
            #return '\n'.join(result)
    #s = gen_html_tree_view_rec(tax.root)
    #return s

def lmdb_reader(lmdb_folder):
    import lmdb
    lmdb_env = lmdb.open(lmdb_folder)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    for key, value in lmdb_cursor:
        yield key, value


def rotate_image(image, rects, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2., h / 2.)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated_im = cv2.warpAffine(image, M, (nW, nH))
    
    rotated_rects = copy.deepcopy(rects)
    for rect in rotated_rects:
        x0, y0, x1, y1 = rect['rect']
        r_x0, r_y0 = np.dot(M, np.asarray([x0, y0, 1]).reshape((3, 1)))
        r_x1, r_y1 = np.dot(M, np.asarray([x1, y1, 1]).reshape((3, 1)))
        r_x2, r_y2 = np.dot(M, np.asarray([x0, y1, 1]).reshape((3, 1)))
        r_x3, r_y3 = np.dot(M, np.asarray([x1, y0, 1]).reshape((3, 1)))
        rect_x0 = min([r_x0, r_x1, r_x2, r_x3])
        rect_y0 = min([r_y0, r_y1, r_y2, r_y3])
        rect_x1 = max([r_x0, r_x1, r_x2, r_x3])
        rect_y1 = max([r_y0, r_y1, r_y2, r_y3])
        rect['rect'] = [rect_x0, rect_y0, rect_x1, rect_y1]

    return rotated_im, rotated_rects

def add_prediction_into_train2(data, predict_file, nms_threshold,
        out_data):
    '''
    this adds the support of different classes
    '''
    from fast_rcnn.nms_wrapper import nms
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
            keep = nms(nms_input, nms_threshold, False, device_id=0)
            rects = [predict_bb[k] for k in keep]
            num_rect_after = len(rects)
            logging.info('{}->{}'.format(num_rect_before, num_rect_after))
            yield row[0], json.dumps(rects), row[-1]

    out_dataset = TSVDataset(out_data)
    if not op.isfile(out_dataset.get_train_tsv()):
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
    return out_data

def add_prediction_into_train(data, predict_file, nms_threshold,
        out_data):
    from fast_rcnn.nms_wrapper import nms
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
            keep = nms(nms_input, nms_threshold, False, device_id=0)
            rects = [predict_bb[k] for k in keep]
            num_rect_after = len(rects)
            logging.info('{}->{}'.format(num_rect_before, num_rect_after))
            yield row[0], json.dumps(rects), row[-1]

    out_dataset = TSVDataset(out_data)
    if not op.isfile(out_dataset.get_train_tsv()):
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
    return out_data

def load_predict(predict_file):
    rows = tsv_reader(predict_file)
    result = {}
    for row in rows:
        image_id = row[0]
        assert image_id not in result
        result[image_id] = json.loads(row[1])
    return result

def load_imagenet_fname_to_url():
    name_urls = tsv_reader('/mnt/sdb/data/raw_data/imagenet/fall11_urls.txt')
    name_to_url = {}
    for x in name_urls:
        if len(x) != 2:
            logging.info('skip: {}'.format('\t'.join(x)))
            continue
        name, url = x
        name_to_url[name] = url
    return name_to_url

def draw_circle(image, point, radius=2, color=[0,0,255]):
    cv2.circle(image, point, radius, color)

def draw_gt(image, label):
    label = label.reshape(-1)
    num_rect = len(label) / 5
    assert num_rect * 5 == len(label)
    rects = []
    txts = []
    im_height, im_width = image.shape[:2]
    for j in range(num_rect):
        if label[j * 5] == 0:
            break
        cx, cy, w, h = label[(j * 5 + 0) : (j * 5 + 4)]
        txt = str(label[j * 5 + 4])
        cx = cx * im_width
        cy = cy * im_height
        w = w * im_width
        h = h * im_height
        rects.append((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 *
            h))
        txts.append(txt)
    draw_bb(image, rects, txts)
    return len(rects)

def draw_grid(grid_im, grid_h_num, grid_w_num):
    height, width = grid_im.shape[:2]
    grid_height = height / grid_h_num
    grid_width = width / grid_w_num
    for _j in range(grid_h_num):
        cv2.line(grid_im, (0, _j * grid_height), (width - 1, _j * grid_height), (0, 0, 255))
    for _i in range(grid_w_num):
        cv2.line(grid_im, (_i * grid_width, 0), (_i * grid_width, height - 1), (0, 0, 255))

def read_blob(fname):
    with open(fname, 'r') as fp:
        s_axis = fp.read(4)
        num_axis = struct.unpack('i', s_axis)[0]
        s_dims = fp.read(4 * num_axis)
        dims = struct.unpack('{}i'.format(num_axis), s_dims)
        count = np.cumprod(dims)[-1]
        s_data = fp.read(4 * count)
        data = np.fromstring(s_data, dtype=np.float32)
        data = data.reshape(dims)
    return data

def url_to_image(url):
    try:
        fp = urllib2.urlopen(url, timeout=5)
        buf = fp.read()
        if fp.geturl() != url:
            logging.info('new url = {}; old = {}'.format(fp.geturl(), url))
            # the image gets redirected, which means the image is not available
            return None
        image = np.asarray(bytearray(buf), dtype='uint8')
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        logging.error("error: {}".format(url))
        return None

def parse_philly_ls_output(output):
    lines = output.split('\n')
    assert len(lines) > 0
    line = lines[0]
    import re
    r = re.match('total ([0-9]*)', line)
    num_rows = int(float(r.groups()[0]))
    all_file = []
    all_dir = []
    for i in range(num_rows):
        line = lines[i + 1]
        p = '(.{1}).*[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}[ ]*(.*)' 
        r = re.match(p, line)
        file_type, file_name = r.groups()
        if file_type == '-':
            all_file.append(file_name)
        else:
            assert file_type == 'd'
            all_dir.append(file_name)
    return all_file, all_dir

def upload_qdoutput(src_path, dest_path, ssh_info):
    # make sure the folder of dest_path in philly exists
    remote_run('mkdir -p {}'.format(dest_path), ssh_info)

    # upload all the files under src_path
    all_src_file = glob.glob(op.join(src_path, '*'))
    for f in all_src_file:
        if op.isfile(f):
            scp(f, dest_path, ssh_info)

    # for the model and the tested files, only upload the best
    all_src_file = glob.glob(op.join(src_path, 'snapshot', 'model_iter_*'))
    all_iter = [parse_iteration2(f) for f in all_src_file]
    max_iters = max(all_iter)
    need_copy_files = [f for f, i in zip(all_src_file, all_iter) if i == max_iters]
    dest_snapshot = op.join(dest_path, 'snapshot')
    remote_run('mkdir -p {}'.format(dest_snapshot), ssh_info)
    for f in need_copy_files:
        scp(f, dest_snapshot, ssh_info)

def philly_upload_qdoutput(src_path, dest_path):
    # make sure the folder of dest_path in philly exists
    philly_mkdir(dest_path)

    # upload all the files under src_path
    all_src_file = glob.glob(op.join(src_path, '*'))
    for f in all_src_file:
        if op.isfile(f):
            philly_upload(f, dest_path)

    # for the model and the tested files, only upload the best
    all_src_file = glob.glob(op.join(src_path, 'snapshot', 'model_iter_*'))
    all_iter = [parse_iteration2(f) for f in all_src_file]
    max_iters = max(all_iter)
    need_copy_files = [f for f, i in zip(all_src_file, all_iter) if i == max_iters]
    dest_snapshot = op.join(dest_path, 'snapshot')
    philly_mkdir(dest_snapshot)
    for f in need_copy_files:
        philly_upload(f, dest_snapshot)

    philly_ls(dest_path)
    philly_ls(op.join(dest_path, 'snapshot/'))


def philly_download_qdoutput(src_path, dest_dir, vc='input'):
    '''
    only download the latest model
    '''
    src_snapshot = op.join(src_path, 'snapshot')
    #output = philly_ls(src_snapshot, vc=vc)
    output = philly_ls(src_snapshot, vc=vc, return_output=True)
    all_file, all_dir = parse_philly_ls_output(output)
    iters = [parse_iteration2(f) for f in all_file]
    max_iters = max(iters)
    need_copy_files = [f for f, i in zip(all_file, iters) if i == max_iters]
    for f in need_copy_files:
        src_file = op.join(src_path, 'snapshot', f)
        dest_folder = op.join(dest_dir, 'snapshot')
        philly_download(src_file, dest_folder, vc)

    # download the source files
    all_file, _ = parse_philly_ls_output(philly_ls(src_path, vc,
        return_output=True))
    for f in all_file:
        src_file = op.join(src_path, f)
        dest_folder = dest_dir
        philly_download(src_file, dest_folder, vc)

def philly_download(src_file, dest_dir, vc='input'):
    dest_dir = op.realpath(dest_dir)
    ensure_directory(dest_dir)
    if vc == 'resrchvc':
        philly_cmd = '/home/jianfw/code/philly/philly_fs/philly-fs.pyc'
        folder_prefix = 'hdfs://gcr/resrchvc/jianfw/'
        cmd = []
        cmd.append(philly_cmd)
        cmd.append('-cp')
        cmd.append('{}{}'.format(folder_prefix, src_file))
        cmd.append('{}'.format(dest_dir))
        cmd_run(cmd, env={'PHILLY_VC': 'resrchvc'})
    else:
        folder_prefix = 'hdfs://{}/{}/'.format('philly-prod-cy4', 
                vc)
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-cp')
        cmd.append('-r')
        cmd.append('{}{}'.format(folder_prefix, src_file))
        cmd.append('{}'.format(op.realpath(dest_dir)))
        philly_input_run(cmd)


def philly_ls(dest_dir, vc='input', return_output=False, cluster='philly-prod-cy4'):
    #cluster = 'gcr' if vc == 'resrchvc' else 'philly-prod-cy4'
    if vc == 'resrchvc':
        folder_prefix = 'hdfs://{}/{}/jianfw/'.format(cluster, vc)
        philly_cmd = '/home/jianfw/code/philly/philly_fs/philly-fs.pyc'
        cmd = []
        cmd.append(philly_cmd)
        cmd.append('-ls')
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        cmd_run(cmd, env={'PHILLY_VC': 'resrchvc'})
    else:
        folder_prefix = 'hdfs://{}/{}/'.format(cluster, vc)
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-ls')
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        output = philly_input_run(cmd, return_output)
        if output:
            logging.info(output)
        return output

def infer_type(vc, cluster):
    all_prem = [('resrchvc', 'gcr'),
            ('pnrsy', 'rr1')]
    all_ap = [('input', 'philly-prod-cy4')]
    all_azure = [('input', 'eu2')]
    if any(v == vc and c == cluster for v, c in all_prem):
        return 'prem'
    elif any(v == vc and c == cluster for v, c in all_ap):
        return 'ap'
    elif any(v == vc and c == cluster for v, c in all_azure):
        return 'azure'
    assert False

def philly_upload_dir(src_dir, dest_dir, vc='input', cluster='philly-prod-cy4'):
    '''
    it looks like the folder upload is not supported. 
    if vc == 'input':
        source ./philly-fs-env.sh jianfw input corpnet
        ./philly-fs -cp -r \
            /home/jianfw/code/quickdetection/data/mturk700_url_as_key \
            hdfs://philly-prod-cy4/input/jianfw/data/qd_data/
    '''
    t = infer_type(vc, cluster)
    if t == 'prem' or t == 'ap':
        disk_type = 'hdfs'
    else:
        disk_type = 'gfs'
    src_dir = op.realpath(src_dir)
    folder_prefix = '{}://{}/{}/'.format(disk_type, cluster, vc)
    if t == 'prem' or t == 'azure':
        philly_cmd = '/mnt/philly-fs/linux/philly-fs'
        cmd = []
        cmd.append(philly_cmd)
        cmd.append('-cp')
        cmd.append('-r')
        cmd.append(src_dir)
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        retry_agent(cmd_run, cmd, env={'PHILLY_VC': vc})
    elif t == 'ap':
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-cp')
        cmd.append('-r')
        cmd.append(src_dir)
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        philly_input_run(cmd)

def retry_agent(func, *args, **kwargs):
    i = 0
    while True:
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.info('fails: try {}-th time'.format(i))
            i = i + 1
            import time
            time.sleep(5)

def philly_input_run(cmd, return_output=False):
    i = 0
    while True:
        try:
            working_dir = op.expanduser('~/code/philly/philly-fs-ap-v5')
            logging.info('working dir: {}'.format(working_dir))
            result = cmd_run(cmd, env={'PHILLY_USER': 'jianfw',
                'PHILLY_VC': 'input',
                'PHILLY_CLUSTER_HDFS_HOST': '131.253.41.35',
                'PHILLY_CLUSTER_HDFS_PORT':
                '81/nn/http/hnn.philly-prod-cy4.cy4.ap.gbl/50070'},
                working_dir=working_dir,
                return_output=return_output)
            logging.info('succeed: {}'.format(' '.join(cmd)))
            return result
        except Exception as e:
            logging.info('fails: try {}-th time'.format(i))
            i = i + 1
            import time
            time.sleep(5)
    
def philly_mkdir(dest_dir, vc='input'):
    '''
    -p is added implicity. 
    '''
    if vc == 'resrchvc':
        philly_cmd = '/home/jianfw/code/philly/philly_fs/philly-fs.pyc'
        folder_prefix = 'hdfs://gcr/resrchvc/jianfw/'
        cmd = []
        cmd.append(philly_cmd)
        cmd.append('-mkdir')
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        cmd_run(cmd, env={'PHILLY_VC': 'resrchvc'})
    else:
        folder_prefix = 'hdfs://{}/{}/'.format('philly-prod-cy4', 
                vc)
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-mkdir')
        #cmd.append('-p') -p is not supported. it is added effectively if not
        #specified
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        philly_input_run(cmd)


def philly_remove(src_folder, vc='input'):
    folder_prefix = 'hdfs://{}/{}/'.format('philly-prod-cy4', 
            vc)
    cmd = []
    cmd.append('./philly-fs')
    cmd.append('-rm')
    cmd.append('-r')
    cmd.append('-f') # no need to confirm if we should remove it. means force
    cmd.append('{}{}'.format(folder_prefix, src_folder))
    philly_input_run(cmd)

def philly_upload(src_file, dest_dir, vc='input'):
    '''
    dest_dir: data/abc
    '''
    if vc == 'resrchvc':
        philly_cmd = '/home/jianfw/code/philly/philly_fs/philly-fs.pyc'
        folder_prefix = 'hdfs://gcr/resrchvc/jianfw/'
        cmd = []
        cmd.append(philly_cmd)
        cmd.append('-cp')
        cmd.append(src_file)
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        cmd_run(cmd, env={'PHILLY_VC': 'resrchvc'})
    elif vc == 'input':
        assert dest_dir.startswith('jianfw')
        src_dir = op.realpath(src_file)
        folder_prefix = 'hdfs://{}/{}/'.format('philly-prod-cy4', 
                vc)
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-cp')
        cmd.append(src_dir)
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        philly_input_run(cmd)

def create_image_by_tile(size, regions):
    '''
    1. create an image with random values
    2. get average width and height of the regions
    3. virtually create a grid on the image. Each grid cell is 2.5 *
    average_height, 2.5 average_width
    4. place a random region inside the cell with a prob of 0.5
    5. if the size is larger than the cell boundary, resize it to the boundary
    if the size is not larger than the cell, randomly select a region to place
    the region there
    '''
    im = np.random.rand(size[0], size[1], 3) * 255.
    im = im.astype(np.uint8)
    
    average_h, average_w = 0, 0
    for region in regions:
        h, w = region.shape[:2]
        average_h = average_h + h
        average_w = average_w + w

    average_h = average_h / len(regions)
    average_w = average_w / len(regions)
    grid_h = int(2.5 * average_h)
    grid_w = int(2.5 * average_w)
    num_h = int(size[0] / grid_h)
    num_w = int(size[1] / grid_w)
    region_idx = np.floor(np.random.rand(num_h, num_w) * len(regions))

    # resize the region if the size is larger than grid
    for i, region in enumerate(regions):
        h, w = region.shape[:2]
        if h > grid_h or w > grid_w:
            ratio = max(1. * grid_h / h, 1. * grid_w / w)
            r2 = cv2.resize(region, (int(ratio * grid_w), int(ratio * grid_h)))
            regions[i] = r2
    
    rects = []
    for h_idx in xrange(num_h):
        for w_idx in xrange(num_w):
            curr_region_idx = int(region_idx[h_idx, w_idx])
            curr_region = regions[curr_region_idx]
            offset_w = int(np.floor(np.random.rand(1) * (grid_w -
                curr_region.shape[1])))
            offset_h = int(np.floor(np.random.rand(1) * (grid_h -
                curr_region.shape[0])))
            origin_x = int(np.floor(w_idx * grid_w))
            origin_y = int(np.floor(h_idx * grid_h))
            start_x = origin_x + offset_w
            start_y = origin_y + offset_h
            curr_region_h, curr_region_w = curr_region.shape[:2]
            im[start_y : (start_y + curr_region_h), 
                    start_x : (start_x + curr_region_w),
                    :] = curr_region
            rect = {'rect': [start_x, start_y, (start_x + curr_region_w),
                (start_y + curr_region_h)]}
            rects.append(rect)
    return rects, im

def dataset_op_tilebb(data):
    source_data = TSVDataset(data)
    dest_name = '{}.tilebb'.format(data)
    dest_data = TSVDataset(dest_name)
    source_train = source_data.get_train_tsv()
    rows = tsv_reader(source_train)
    size_regions = []
    for row in rows:
        rects = json.loads(row[1])
        im = img_from_base64(row[-1])
        regions = []
        for rect in rects:
            rect_class = rect['class']
            x0, y0, x1, y1 = [int(r) for r in rect['rect']]
            # if the region is close to the boundary, let's not use it.
            if x0 < im.shape[1] * 0.05:
                continue
            if y0 < im.shape[0] * 0.05:
                continue
            if x1 > im.shape[1] * 0.95:
                continue
            if y1 > im.shape[0] * 0.95:
                continue
            region = im[y0:y1, x0:x1, :]
            regions.append((region, rect_class))
        size = im.shape[:2]
        size_regions.append((size, regions))
    
    # for each bounding box, tile it 
    def gen_rows():
        idx = 0
        for size, regions in size_regions:
            for region, rect_class in regions:
                rects, im = create_image_by_tile(size, [region])
                for r in rects:
                    r['class'] = rect_class
                yield str(idx), json.dumps(rects), encoded_from_img(im)
                idx = idx + 1
    tsv_writer(gen_rows(), dest_data.get_train_tsv())

    source_test = source_data.get_data('test')
    dest_test = dest_data.get_data('test')
    if op.islink(dest_test):
        os.remove(dest_test)
    os.symlink(op.relpath(source_test, op.dirname(dest_test)), dest_test)
    populate_dataset_details(dest_name)

def dataset_op_removelabel(data):
    source = TSVDataset(data)
    dest = TSVDataset('{}_removelabel'.format(data))
    rows = tsv_reader(source.get_train_tsv())
    def gen_rows():
        for row in rows:
            row[1] = json.dumps([])
            yield row
    tsv_writer(gen_rows(), dest.get_train_tsv())

def sample_labels_by_label(rects, select_bb):
    num = len(rects)
    idx = range(num)
    random.shuffle(idx)
    label_to_quota = {}
    for r in rects:
        label_to_quota[r['class']] = select_bb
    result = []
    for i in idx:
        r = rects[i]
        if label_to_quota[r['class']] > 0:
            result.append(r)
            label_to_quota[r['class']] = label_to_quota[r['class']] - 1
    return result

def sample_labels(rects, select_bb):
    if len(rects) <= select_bb:
        return rects
    num = len(rects)
    idx = range(num)
    random.shuffle(idx)
    result = []
    for i in idx[:select_bb]:
        result.append(rects[i])
    return result


def dataset_op_select_by_label(data, select_image, select_bb, out_data):
    source_data = TSVDataset(data)
    dest_data = TSVDataset(out_data)
    
    source_num_training = source_data.get_num_train_image()
    random.seed(777)
    seq = range(source_num_training)
    random.shuffle(seq)
    select_image = min(select_image, len(seq))
    select_idx = seq[:select_image]
    source_train = TSVFile(source_data.get_data('train'))
    def gen_rows():
        for idx in select_idx:
            row = source_train.seek(idx)
            rects = json.loads(row[1])
            rects = sample_labels_by_label(rects, select_bb)
            row[1] = json.dumps(rects)
            yield row
    def gen_full_rows():
        for idx in select_idx:
            row = source_train.seek(idx)
            yield row

    if op.isfile(dest_data.get_train_tsv()):
        logging.info('ignore to generate {} since it exists'.
                format(dest_data.get_train_tsv()))
    else:
        tsv_writer(gen_rows(), dest_data.get_train_tsv())
    shutil.copy(source_data.get_labelmap_file(), 
            dest_data.get_labelmap_file())
    dest_test = dest_data.get_data('test')
    tsv_writer(gen_full_rows(), dest_test)
    populate_dataset_details(out_data)

def dataset_op_select(data, select_image, select_bb, keep_others,
        out_data):
    source_data = TSVDataset(data)
    dest_data = TSVDataset(out_data)
    
    source_num_training = source_data.get_num_train_image()
    random.seed(777)
    seq = range(source_num_training)
    random.shuffle(seq)
    select_image = min(select_image, len(seq))
    select_idx = seq[:select_image]
    source_train = TSVFile(source_data.get_data('train'))
    def gen_rows():
        for idx in select_idx:
            row = source_train.seek(idx)
            rects = json.loads(row[1])
            rects = sample_labels(rects, select_bb)
            row[1] = json.dumps(rects)
            yield row
    def gen_full_rows():
        for idx in select_idx:
            row = source_train.seek(idx)
            yield row

    if op.isfile(dest_data.get_train_tsv()):
        logging.info('ignore to generate {} since it exists'.
                format(dest_data.get_train_tsv()))
    else:
        tsv_writer(gen_rows(), dest_data.get_train_tsv())
    shutil.copy(source_data.get_labelmap_file(), 
            dest_data.get_labelmap_file())
    dest_test = dest_data.get_data('test')
    tsv_writer(gen_full_rows(), dest_test)
    populate_dataset_details(out_data)

def load_labelmap(data):
    import tsv_io
    return tsv_io.load_labelmap(data)
    #dataset = TSVDataset(data)
    #return dataset.load_labelmap()

def parse_data(full_expid):
    import qd_common
    return qd_common.parse_data(full_expid)
    #all_data = os.listdir('data/')
    #candidates = [data for data in all_data if full_expid.startswith(data)]
    #max_length = max([len(c) for c in candidates])
    #return [c for c in candidates if len(c) == max_length][0]

def parse_test_data(predict_file):
    import qd_common
    return qd_common.parse_test_data(predict_file)
    #all_data = os.listdir('data/')
    #candidates = [data for data in all_data if '.caffemodel.' + data in predict_file]
    #assert len(candidates) > 0
    #max_length = max([len(c) for c in candidates])
    #return [c for c in candidates if len(c) == max_length][0]

def parse_iteration2(file_name):
    r = re.match('.*model_iter_([0-9]*)\..*', file_name)
    return int(float(r.groups()[0]))

def parse_iteration(predict_file):
    import qd_common
    return qd_common.parse_iteration(predict_file)

def get_all_predict_files(full_expid):
    import qd_common
    return qd_common.get_all_predict_files(full_expid)

if __name__ == '__main__':
    from qd_common import init_logging
    init_logging()
    test_update_yolo_test_proto()
    test_visualize()
    #test_central_overlap()
    #test_build_tree_from_tree_file()
    #untar_imagenet22()
    #untar_imagenet3k()
    #test_merge_labelset()
    #bing2k()
    #test_mapper()
    #study_imagenet_url()
    #build_taxonomy()
    #study_imagenet3k()
    #create_imagenet_tsv()
    #imagenet_label_to_noffset()
    #noffset_to_9000_idx()
    #test_labels2noffsets()
    #test_noffset_imagenet()
    #test_gen_term_list()
    #test_gen_cls_specific_th()
    #test_gen_noffset()

