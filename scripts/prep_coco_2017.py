from random import shuffle
import logging
import os, os.path as op
import sys;
import json
import base64
import cv2
from process_image import draw_bb, show_image
from process_tsv import tsv_writer
from qd_common import init_logging


def prep_coco():
    pass

def test_prep_coco():
    coco_root = '/raid/jianfw/data/raw_coco/'
    ann_folder = op.join(coco_root, "annotations");
    truthlocs = [('instances_train2017.json','train2017')];
    truthlocs = [('instances_val2017.json','val2017')];
    out_folder = '/raid/jianfw/data/coco2017'

    for datasets in truthlocs:
        annfile = op.join(ann_folder, datasets[0]);
        imgfolder = op.join(coco_root, datasets[1]);
        with open(annfile,'r') as jsin:
            print("Loading annotations...")        
            truths = json.load(jsin)
            #map id to filename
            imgdict = {x['id']:x['file_name'] for x in truths['images']};
            catdict = {x['id']:x['name'] for x in truths['categories']};
            anndict = { x:[] for x in imgdict };
            for ann in truths['annotations']:
                imgid = ann['image_id'];
                bbox = ann['bbox'];
                bbox[2] += bbox[0]-1;
                bbox[3] += bbox[1]-1;
                cid = ann['category_id'];
                crect = {'class':catdict[cid], 'rect':bbox}
                anndict[imgid]+=[crect];
        
        cnames=sorted(catdict.values());
        with open("labelmap.txt","w") as tsvout:
            for cname in cnames:
                tsvout.write(cname+"\n")
        print("Saving tsv...")

        def generate_tsv_row():
            image_ids = anndict.keys()
            logging.info('shuffle the list ({})'.format(len(image_ids)))
            shuffle(image_ids)

            for i, image_id in enumerate(image_ids):
                imgf = op.join(imgfolder,imgdict[image_id]);
                im = cv2.imread(imgf)
                if im is None:
                    logging.info('{} is not decodable'.format(imgf))
                    continue
                with open(imgf, 'rb') as f:
                    image_data = base64.b64encode(f.read());
                if (i % 100) == 0:
                    logging.info(i)
                yield str(image_id), json.dumps(anndict[image_id]), image_data

        tsv_writer(generate_tsv_row(), op.join(out_folder, datasets[1] +
            '.tsv'))

if __name__ == '__main__':
    init_logging()
    test_prep_coco()
    
