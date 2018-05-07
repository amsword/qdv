from yolotrain import CaffeWrapper
from pprint import pprint
import json
import os
import matplotlib.pyplot as plt
import numpy as np

cache_file = 'cache_result.json'
expids = {'baseline':'baseline', 
        'remove_aeroplane10':'box10', 
        'remove_aeroplane_image10':'image10',
        'remove_aeroplane25':'box25', 
        'remove_aeroplane_image25':'image25',
        'remove_aeroplane50':'box50',
        'remove_aeroplane_image50':'image50',
        'remove_aeroplane75':'box75', 
        'remove_aeroplane_image75':'image75',
        'remove_aeroplane80':'box80', 
        'remove_aeroplane_image80':'image80', 
        'remove_aeroplane90':'box90', 
        'remove_aeroplane_image90':'image90', 
        'remove_aeroplane':'box100'}

table_rows = ['baseline', 'box10', 'image10', 'box25', 'image25',
        'box50', 'image50', 'box75', 'image75', 'box80', 'image80',
        'box90', 'image90',
        'box100']  

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
    
    c = CaffeWrapper()
    
    result = {}
    for expid in expids.keys():
        kwargs['expid'] = expid 
        is_unfinished, s = c.summarize(**kwargs)
        xs, ys, best_model_iter, best_model_class_ap, model_dir = s
        result[expid] = best_model_class_ap
    with open(cache_file, 'w') as fp:
        json.dump(result, fp)

def print_table(r, expids, labels):
    lines = []
    line = '\\begin{table}'
    lines.append(line)
    line = '\\begin{{tabular}}{{{}@{{}}}}'.format('@{~}c' * (len(labels) + 1))
    lines.append(line)
    lines.append('\\toprule')
    lines.append('& {}\\\\'.format(' & '.join(labels)))
    for expid in expids: 
        line = '{} & {}\\\\'.format(expid, 
                ' & '.join(('{0:.2f}'.format(r[expid][l]) for l in labels)))
        lines.append(line)
    lines.append('\\bottomrule')
    line = '\\end{tabular}'
    lines.append(line)
    line = '\\end{table}'
    lines.append(line)
    print '\n'.join(lines)


labels = None
r = {}
for expid in expids:
    x = expids[expid]
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

labels.remove('aeroplane')
labels = ['aeroplane'] + labels
print_table(r, labels, table_rows)
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

