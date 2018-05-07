import re
import fnmatch
import glob
import os.path as op
import time
import os
import logging
from qd_common import init_logging
from pprint import pformat

def collect():
    pass

def fine_to_remove(s, expiration_days):
    curr_time = time.time()
    e = expiration_days*24*60*60
    return curr_time-s.st_atime > e and \
            curr_time - s.st_mtime > e

def test_collector():
    logging.info('start')
    total_size = 0
    all_data = []
    #folder = '/vighd/dlworkspace/work/jianfw/work/qd_output/'
    folder = 'output/'
    t = 0
    for aname in iter_to_be_deleted2(folder):
        if aname.endswith('.simple'):
            logging.info('skip: {}'.format(aname))
            continue
        s = os.stat(aname)
        if s.st_size < 1 * 1024 * 1024:
            logging.info('too small: skipping {}'.format(aname))
            continue
        total_size = total_size + s.st_size
        logging.info('removing {} - {}'.format(aname,
            total_size/1024./1024./1024.))
        os.remove(aname)
    logging.info(total_size / 1024.0 / 1024.0 / 1024.0)

def parse_iter(file_name):
    m = re.match('.*model_iter_([0-9]*)[\.nobn]*\.[caffemodel|solverstate].*', file_name)
    if m is None:
        return None
    else:
        return int(m.groups()[0])

def iter_to_be_deleted2(folder):
    iter_extract_pattern = '.*model_iter_([0-9]*)\..*'
    must_have_in_folder = ['CARPK']
    for root, dirnames, file_names in os.walk(folder):
        matched = False
        for f in must_have_in_folder:
            if f in root:
                matched = True
                break
        if not matched:
            continue
        ms = [(f, re.match(iter_extract_pattern, f)) for f in file_names]
        ms = [(f, int(m.groups()[0])) for f, m in ms if m]
        if len(ms) == 0:
            continue
        max_iter = max(x[1] for x in ms)
        to = [f for f, i in ms if i < max_iter]
        if len(to) == 0:
            continue
        for t in to:
            yield os.path.join(root, t)

#def iter_to_be_deleted():
    #folder = '/vighd/dlworkspace/work/jianfw/work/qd_output'
    #white_folder_list = ['/vighd/dlworkspace/work/jianfw/.vim']
    #safe_remove_file_pattern = ['.*\.caffemodel\.report', 
            #'.*model_iter_([0-9]*)\.caffemodel\.eval']
    #must_has_keyword = ['voc']
    ##safe_remove_file_pattern = ['.*\.solverstate']
    #expiration_days = 1
    #for root, dirnames, file_names in os.walk(folder):
        #if root in white_folder_list:
            #continue
        #if any(w in root for w in white_folder_list):
            #continue
        #proceed = False
        #for k in must_has_keyword:
            #if k in root:
                #proceed = True
                #break
        #if not proceed:
            #continue
        #if 'snapshot' in root:
            #anames = [op.join(root, f) for f in file_names if f != 'map.png']
            #x = [(aname, parse_iter(aname)) for aname in anames]
            #y = [(f, n) for f, n in x if n is None]
            #if len(y) > 0: 
                #logging.info(y)
                #assert False
            #if len(x) > 0:
                #max_num = max(x, key=lambda p: p[1])[1]
                #x = [(aname, num) for aname, num in x if num < max_num]
                #for aname , num in x:
                    #s = os.stat(aname)
                    #yield aname, s.st_size
        #else:
            #for file_name in file_names:
                #aname = op.join(root, file_name)
                #if op.isdir(aname):
                    #continue
                #if not op.isfile(aname):
                    #continue
                #if not any(re.match(p, file_name) is not None for p in
                        #safe_remove_file_pattern):
                    #continue
                #s = os.stat(aname)
                #if fine_to_remove(s, expiration_days):
                    #yield aname, s.st_size

if __name__ == '__main__':
    init_logging()
    test_collector()
    #iter_to_be_deleted2()

