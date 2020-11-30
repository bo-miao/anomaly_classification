from utils import lr_scheduler, metric, prefetch, summary
import os, sys
import time
import numpy as np
from collections import OrderedDict
import glob
import math
import copy
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

rng = np.random.RandomState(2020)


def get_the_number_of_params(model, is_trainable=False):
    """get the number of the model"""
    if is_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def AUC(anomal_scores, labels):
    frame_auc = 0
    try:
        frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    except:
        print("AUC Cal ERROR: ", labels, anomal_scores)
    
    return frame_auc


def evaluate_resnet(model, test_batch, args):
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), single_time, prefix="Evaluation: ")

    model.eval()

    counter = 0
    tp = 0
    for k, (images, labels) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        label = labels if args.label else None
        label = label.view(-1)
        input_image = images.detach()

        a = time.time()
        with autocast():
            logit = model.forward(input_image)
        if args.evaluate_time:
            single_time.update((time.time() - a) * 1000)
            progress.print(counter)
            print("Single batch time cost {}ms".format(1000 * (time.time() - a)))

        class_vector = F.softmax(logit, 1).data.squeeze()
        assert len(class_vector) == len(label), "class number must match"
        probs, idx = class_vector.sort(1, True)
        idx = idx[:,0]
        tp += torch.sum(idx.view(-1)==label).item()
        counter += len(label)

    accuracy = tp / counter
    print("INFERENCE ACCURACY IS {}".format(accuracy))
    return accuracy


def visualize(recon, gt):
    b, c, h, w = recon.size()
    for i in range(b):
        img1, img2 = recon[i], gt[i]
        img = torch.cat((img1, img2), dim=2)
        img = 255. * (img + 1.) / 2.
        img = img.squeeze(0).byte().cpu().numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (600, 300))
        frame, name = img, str(int(time.time()*1000))
        cv2.imwrite(os.path.join("/data/miaobo/tmp", name+".jpg"), frame)

    return True


def visualize_single(image):
    b, c, h, w = image.size()
    for i in range(b):
        img = image[i]
        img = 255. * (img + 1.) / 2.
        img = img.byte().cpu().numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame, name = img, str(int(time.time()*1000))
        cv2.imwrite(os.path.join("/data/miaobo/tmp", name+".jpg"), frame)

    return True
