#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Harvard Spring 2018 AC297R BASF Group, based on code from Xinlei Chen, Ross Girshick
# --------------------------------------------------------

"""
Script for inference detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import os, cv2
import argparse
from glob import glob
import errno

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLS_NOT_INTERESTED = 'not_interested'
CLS_INTERESTED = 'interested'

CLASSES = ('__background__',
           CLS_NOT_INTERESTED, CLS_INTERESTED)

NETS = {'vgg16': ('vgg16.pth',),'res101': ('res101.pth',),'res152':('res152.pth',)}

HOME_DIR = os.path.expanduser("~")
MODEL_DIR = os.path.join(HOME_DIR,"models")
DATA_DIR = os.path.join(HOME_DIR,"imgwork")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def output_result(image_name, im, result, thresh=0.5):
    """Draw detected bounding boxes."""
    makedirs(os.path.join(OUTPUT_DIR, "labeled_images"))
    makedirs(os.path.join(OUTPUT_DIR, "boxes"))

    label_file = open(os.path.join(OUTPUT_DIR, "boxes", "{}.txt".format(image_name)), "w")

    fig = plt.gcf()

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    colors = {CLS_NOT_INTERESTED:"blue", CLS_INTERESTED:"red"}
    print("cls, cood1, cood2, cood3, cood4, conf", file=label_file)
    for class_name in result:
        dets = result[class_name]
        inds = np.where(dets[:, -1] >= thresh)[0]
    	for bb in dets:
    	    print(class_name, ",", ", ".join([str(b) for b in bb]), file=label_file)
	if len(inds) == 0:
            continue
	for i in inds:
	    bbox = dets[i, :4]
	    score = dets[i, -1]

	    ax.add_patch(
	        plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[class_name], linewidth=3.5)
	        )
	    ax.text(bbox[0], bbox[1] - 2,
	            '{:s} {:.3f}'.format(class_name, score),
	            bbox=dict(facecolor=colors[class_name], alpha=0.5),
	            fontsize=14, color='white')

    ax.set_title('RED:{},BLUE:{},THRES:{}'.format(CLS_INTERESTED,CLS_NOT_INTERESTED,thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(OUTPUT_DIR, "labeled_images", "{}.png".format(image_name)))
    label_file.close()

def run(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(INPUT_DIR, image_name)
    print(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.7
    result = {}
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets).cuda(), NMS_THRESH)
        dets = dets[keep.cpu().numpy(), :]
        result[cls] = dets
    output_result(image_name, im, result,CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = (832,)
    cfg.TEST.MAX_SIZE = 1128
    args = parse_args()

    # model path
    demonet = args.demo_net
    saved_model = os.path.join(HOME_DIR,"models",NETS[demonet][0])

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture(3,
                          tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model))
    net.cuda()

    net.eval()

    print('Loaded network {:s}'.format(saved_model))

    im_names = sorted([os.path.basename(p) for p in glob(os.path.join(INPUT_DIR, "*"))])
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Analyzing {}/{}'.format(INPUT_DIR, im_name))
        run(net, im_name)
