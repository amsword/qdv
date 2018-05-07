import _init_paths
from tsv_io import tsv_reader
from qd_common import write_to_file
from qd_common import init_logging
from qd_common import worth_create
import logging
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_to_cocoformat(predict_tsv, predict_json, label_to_id):
    # read all the data and convert them into coco's json format
    rows = tsv_reader(predict_tsv)
    annotations = []
    image_names = []
    image_ids = []
    for row in rows:
        image_id = int(row[0])
        rects = json.loads(row[1])
        for rect in rects:
            r = rect['rect']
            ann = [r[0], r[1], r[2] - r[0], r[3] - r[1]]
            category_id = label_to_id[rect['class']]
            annotations.append({'image_id': image_id,
                'category_id': category_id,
                'bbox': ann,
                'score': rect['conf']})
    write_to_file(json.dumps(annotations),
            predict_json)

def coco_eval(predict_tsv, gt_json):
    predict_json = predict_tsv + '.coco.json'
    # gt file
    with open(gt_json, 'r') as fp:
        gt = json.load(fp)

    if worth_create(predict_tsv, predict_json):
        logging.info('create json file: {}'.format(predict_json))
        label_to_id = {cat_info['name']: cat_info['id'] for cat_info in gt['categories']}
        convert_to_cocoformat(predict_tsv, predict_json, label_to_id)
    else:
        logging.info('ignore to create the json file')

    cocoGt=COCO(gt_json)
    cocoDt=cocoGt.loadRes(predict_json)

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    results = cocoEval.summarize()

    return '\n'.join(results)

def test_coco_eval():
    #predict_tsv = './output/coco2017_darknet19_A/snapshot/model_iter_236544.caffemodel.coco2017.predict'
    predict_tsv = \
    './output/coco2017_resnet101_A/snapshot/model_iter_236544.caffemodel.coco2017.mainTrainRatio.testInput416.608.992.1024.predict'
    predict_tsv = \
    './output/coco2017_darknet19_A/snapshot/model_iter_236544.caffemodel.coco2017.predict'
    gt_json = '/gpu02_raid/jianfw/code/cocoapi/annotations/instances_val2017.json'
    coco_eval(predict_tsv, gt_json)
    

if __name__ == '__main__':
    init_logging()
    test_coco_eval()

