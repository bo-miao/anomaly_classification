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


def rel2abs(box, h, w):
    box[:, 0] *= w
    box[:, 2] *= w
    box[:, 1] *= h
    box[:, 3] *= h
    return box


# label resize & loss support
def get_object_images(images, labels, bboxes, args):
    b, c, h, w = images.size()
    bbox_num = []
    bbox_mask = []
    new_bboxes = []
    for i, bbox in enumerate(bboxes):
        bbox_num.append(bbox.size()[0])
        if bbox.size()[0] > 0:
            bbox_mask.append(i)
            bbox = rel2abs(bbox, h, w)  # res to abs coord
            new_bboxes.append(bbox)  # non empty boxes

    if len(new_bboxes) == 0:  # prevent all miss objects
        patches = None
        new_labels = None
        return patches, new_labels, bbox_num

    new_images = images[bbox_mask]  # images with non empty boxes
    patch_h, patch_w = args.path_h, args.path_w  # TODO: Alignsize
    patches = torchvision.ops.roi_align(new_images, new_bboxes, output_size=(patch_h, patch_w))
    patches = patches.view(-1, c, patch_h, patch_w)
    assert patches.size()[0] == sum(bbox_num), "patch number does not match bbox_num"
    new_labels = torch.zeros(sum(bbox_num)).cuda(non_blocking=True)
    start_ = 0
    for i, label in enumerate(labels):
        new_labels[start_:start_ + bbox_num[i]] = label
        start_ += bbox_num[i]

    return patches, new_labels, bbox_num  # [K,C,H,W] [K] [B]


def get_the_number_of_params(model, is_trainable=False):
    """get the number of the model"""
    if is_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def psnr(mse):
    e = 1e-8
    return 10 * math.log10(1 / (mse+e))


def normalize_psnr_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def psnr_score_list(psnr_list):
    psnr_score_list = list()
    print("PSNR MAX MIN: ", np.max(psnr_list), " AND ", np.min(psnr_list))
    for i in range(len(psnr_list)):
        psnr_score_list.append(normalize_psnr_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return psnr_score_list


def AUC(anomal_scores, labels):
    frame_auc = 0
    try:
        frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    except:
        print("AUC Cal ERROR: ", labels, anomal_scores)
    
    return frame_auc


def plot_AUC(anomal_scores, labels):
    try:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(np.squeeze(labels, axis=0), np.squeeze(anomal_scores))
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('AUC')
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig('/home/miaobo/project/anomaly_demo2/ckpt/'+str(time.strftime("%m%d%H%M",time.localtime()))+'.png')
    except:
        print("PLOTING ROC CURVE ERROR")


def score_sum_single(list1):
    list_result = []
    for i in range(len(list1)):
        list_result.append((list1[i]))
    return list_result


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


def evaluate_memory(model, test_batch, args, memory):
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), avg_loss, single_time, prefix="Evaluation: ")
    model.eval()

    label_list = []
    psnr_list = []
    counter = 0
    for k, (images, labels) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        counter += 1

        label = labels if args.label else None
        channel = (images.size()[1] // args.c - 1) * args.c
        # input_image = images[:, 0:channel]
        # target_image = images[:, channel:]
        input_image = images.detach()
        target_image = images[:, channel:]
        b,c,h,w = images.shape

        a = time.time()
        with autocast():
            reconstructed_image, loss, _ = model.forward(input_image, gt=target_image, memory=memory, train=False)
            loss = 0.8 * loss['pixel_loss'].view(b,-1).mean(1) + 0.2 * loss['memory_loss'].view(b,-1).mean(1)
        if args.evaluate_time:
            single_time.update((time.time() - a)*1000)
            progress.print(counter)
            print("Single batch time cost {}ms, loss {}".format(1000*(time.time()-a), loss.mean().item()))

        assert len(loss) == len(label), "During inference, loss sample number must match label sample number."
        for i in range(len(loss)):
            mse_reconstruction = loss[i].item()
            psnr_list.append(psnr(mse_reconstruction))
            label_list = np.append(label_list, label[i].item())
            avg_loss.update(loss[i].item(), 1)

    psnr_score_total_list = np.asarray(psnr_score_list(psnr_list))
    assert psnr_score_total_list.size == label_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."
    accuracy = roc_auc_score(y_true=label_list, y_score=1-psnr_score_total_list)
    # plot_AUC(psnr_score_total_list, np.expand_dims(1 - labels_list, 0))
    print("EVALUATE FRAME NUMBER: ", psnr_score_total_list.size)
    return accuracy, avg_loss.avg


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
