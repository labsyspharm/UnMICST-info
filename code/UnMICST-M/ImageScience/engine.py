import math
import sys
import time
import torch
import pickle

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

import pdb
import os
from torchvision.utils import save_image
import numpy as np


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch,
                    iter_count, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if iter_count >= 1000:
            lr_scheduler = None
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iter_count += 1

    return iter_count, metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
    iou_types.append("segm")
    if isinstance(model_without_ddp,
                  torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, is_test=False, is_vis=False, 
    draw_bbox=False, vis_dir='./vis_results', cpu=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    res_list = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if cpu == False:
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)
        res_list.append(res)

        # visualization
        if is_test:
            if os.path.exists(vis_dir) == False:
                os.makedirs(vis_dir)
            with open(os.path.join(vis_dir, 'pred_res.pkl'), 'wb') as pkl_file:
                pickle.dump(res_list, pkl_file)
            if is_vis:
                coco_evaluator.visualize(res, images, vis_dir, draw_bbox)
                input('wait key...')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    mAP_scores = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return mAP_scores


@torch.no_grad()
def test(model, data_loader, device, is_vis=False, draw_bbox=False,
         vis_dir='./vis_results', overlap_ratio=0.3, patch_size=128, cpu=False):
    # 64 performs well
    # 128 performs the best
    # 256, 512 doesn't work
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for img, img_path in metric_logger.log_every(data_loader, 100, header):
        dataset_path = os.path.join('results',
                                    img_path[0].split('/cycif/')[1].split('/dearray/')[0])
        img_name = img_path[0].split(
            '/cycif/')[1].split('/dearray/')[1].split('.tif')[0]
        img = img.to(device)
        if cpu == False:
            torch.cuda.synchronize()
        img_h = img.shape[2]
        img_w = img.shape[3]
        slide_len = int(patch_size*(1.-overlap_ratio))
        y_list = np.arange(0, img_h, slide_len)
        x_list = np.arange(0, img_w, slide_len)
        whole_segm = np.zeros((img_h, img_w), dtype=np.uint32)
        seen_segm = np.zeros((img_h, img_w), dtype=np.uint32)
        cell_id = 1
        score_dict = {}
        for y_id in y_list:
            y_max = min(img_h, y_id + patch_size)
            y_min = y_id
            if y_max == img_h:
                y_min = img_h - patch_size
            for x_id in x_list:
                x_max = min(img_w, x_id + patch_size)
                x_min = x_id
                if x_max == img_w:
                    x_min = img_w - patch_size
                outputs = model(img[..., y_min:y_max, x_min:x_max])
                output = {k: v.to(cpu_device) for k, v in outputs[0].items()}
                if len(output['scores']) == 0:
                    continue
                # pdb.set_trace()
                for out_box, out_label, out_score, out_mask in zip(
                        output['boxes'], output['labels'], output['scores'], output['masks']):
                    # if out_score < 0.3:
                    #     continue
                    # record result
                    segm_patch = whole_segm[y_min:y_max, x_min:x_max]
                    trg_mask = (out_mask[0] > 0.5).numpy()
                    # check overlap with existing cells
                    unique_list = np.unique(segm_patch[trg_mask])
                    is_overlap = False
                    for exist_id in unique_list:
                        if exist_id == 0:
                            continue
                        src_mask = (segm_patch == exist_id)
                        iou_score = np.sum(src_mask & trg_mask) / \
                            np.sum(src_mask | trg_mask)
                        if iou_score > 0.5:
                            segm_patch[trg_mask] = exist_id
                            score_dict[exist_id] = max(
                                score_dict[exist_id], out_score)
                            is_overlap = True
                            break
                    if is_overlap == False:
                        segm_patch[trg_mask & (segm_patch == 0)] = cell_id
                        for exist_id in unique_list:
                            if exist_id == 0:
                                continue
                            if out_score > score_dict[exist_id]:
                                src_mask = (segm_patch == exist_id)
                                segm_patch[trg_mask & src_mask] = cell_id
                        # segm_patch[trg_mask] = cell_id
                        score_dict[cell_id] = out_score
                        cell_id += 1
        if os.path.exists(dataset_path) == False:
            os.makedirs(dataset_path)
        res_dict = {'segm': whole_segm, 'score': score_dict}
        with open(os.path.join(dataset_path, '{}.pkl'.format(img_name)), 'wb') as pkl_file:
            pickle.dump(res_dict, pkl_file)
#         with open('results/')
        # visualization
#         if is_vis:
#             if os.path.exists(vis_dir) == False:
#                 os.makedirs(vis_dir)
#             coco_evaluator.visualize(res, images, vis_dir, draw_bbox)

    # with open(os.path.join(vis_dir, 'pred_res.pkl'), 'wb') as pkl_file:
    #     pickle.dump(res_list, pkl_file)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # coco_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    # torch.set_num_threads(n_threads)
    return 0
