import copy
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from coco import CocoDetection as CocoDet

import transforms as T
import torchvision.transforms as visionT
import pdb
import numpy as np
import cv2

import glob

import random
import tifffile


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        bg_maps = []
        for obj in anno:
            bg_map = coco_mask.decode(obj['bg_map'])
            bg_map = torch.as_tensor(bg_map, dtype=torch.uint8)
            bg_maps.append(bg_map)
        bg_maps = torch.stack(bg_maps, dim=0)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        bg_maps = bg_maps[keep]

        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["bg_maps"] = bg_maps
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
        # Add for ignore map
        # target["segmentation"] =
        # 'bg_map'
        # segmentations = [obj["bg_map"] for obj in anno]

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    # assert isinstance(dataset, torchvision.datasets.CocoDetection)
    assert isinstance(dataset, CocoDet)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        if isinstance(dataset, CocoDet):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    # if isinstance(dataset, torchvision.datasets.CocoDetection):
    if isinstance(dataset, CocoDet):
        return dataset.coco
    return convert_to_coco_api(dataset)


# class CocoDetection(torchvision.datasets.CocoDetection):
class CocoDetection(CocoDet):
    def __init__(self, img_folder, ann_file, transforms, use_channel='dapi', augmentation='none', train=True, testdomain='clean'):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            use_channel=use_channel,
                                            augmentation=augmentation,
                                            train=train,
                                            testdomain=testdomain)
        self._transforms = transforms
        self.train = train

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        # print(type(target['annotations']))
        # print(target['annotations'][0].keys())
        if self._transforms is not None:
            if type(img) == list:
                rand_seed = random.randint(0, 1251216)
                random.seed(rand_seed)
                dapi_img, _ = self._transforms(img[0], target)
                random.seed(rand_seed)
                lamin_img, target = self._transforms(img[1], target)
                # First two: DAPI, Third: Lamin
                img = torch.cat([dapi_img[:2],
                                 lamin_img[0].unsqueeze(0)], dim=0)
            else:
                img, target = self._transforms(img, target)
            # img, target = self._transforms(img, target)
        # print(target.keys())
        # print(target['annotations'][0].keys())
        return img, target
            # image_id = self.ids[idx]
            # target = dict(image_id=image_id, annotations=target)
            # # print(type(target['annotations']))
            # # print(target['annotations'][0].keys())
            # if self._transforms is not None:
            #     for d_id, (img_name, img) in enumerate(imgs.items()):
            #         if type(img) == list:
            #             rand_seed = random.randint(0, 1251216)
            #             random.seed(rand_seed)
            #             dapi_img, _ = self._transforms(img[0], target)
            #             random.seed(rand_seed)
            #             lamin_img, target = self._transforms(img[1], target)
            #             # First two: DAPI, Third: Lamin
            #             imgs[img_name] = torch.cat(
            #                 [dapi_img[:2], lamin_img[0].unsqueeze(0)], dim=0)
            #         else:
            #             if d_id == (len(imgs)-1):
            #                 imgs[img_name], target = self._transforms(img,
            #                                                           target)
            #             else:
            #                 imgs[img_name], _ = self._transforms(img,
            #                                                      target)
            #     # img, target = self._transforms(img, target)
            # # print(target.keys())
            # # print(target['annotations'][0].keys())
            # return imgs, target


def get_coco(root, image_set, transforms, mode='instances'):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_coco_kp(root, image_set, transforms):
    return get_coco(root, image_set, transforms, mode="person_keypoints")


def get_cellsegm(root_path, process_phase, transform_func, mode='instances',
                 use_channel='dapi', augmentation='none', train=True,
                 testdomain='clean'):
    t = [ConvertCocoPolysToMask()]
    if transform_func is not None:
        t.append(transform_func)
    transforms = T.Compose(t)
    img_folder = os.path.join(root_path, process_phase)
    ann_file = os.path.join(root_path,
                            'coco_cellsegm_{}.json'.format(process_phase))
    dataset = CocoDetection(img_folder,
                            ann_file,
                            transforms=transforms,
                            use_channel=use_channel,
                            augmentation=augmentation,
                            train=train,
                            testdomain=testdomain)
    if process_phase == "train":
        dataset = _coco_remove_images_without_annotations(dataset)
    return dataset


class TestDetection(object):
    def __init__(self, img_folder, use_channel='dapi'):
        self.use_channel = use_channel
        self.img_paths = sorted(glob.glob(os.path.join(img_folder, '*.tif')))
        # self.img_paths = [os.path.join(img_folder, '101.tif')]
        self.transform = visionT.ToTensor()

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        np_img = tifffile.imread(img_path)
        np_img = (np_img.astype(np.float)/255.).astype(np.uint8)
        pil_img = Image.fromarray(np_img).convert('RGB')
        tensor_img = self.transform(pil_img)
        '''
        Gamma correction
        1./2.2 was too much 
        1./1.5 was good
        '''
        tensor_img = torch.pow(tensor_img, 1./1.5)
#         if self.use_channel == 'dapi':
#             img = Image.fromarray(np_img[0]).convert('RGB')
#         elif self.use_channel == 'lamin':
#             img = Image.fromarray(np_img[5]).convert('RGB')
#         elif self.use_channel == 'both':
#             dapi_img = Image.fromarray(np_img[0]).convert('RGB')
#             lamin_img = Image.fromarray(np_img[5]).convert('RGB')
#             img = [dapi_img, lamin_img]
#             # First two: DAPI, Third: Lamin
#             img = torch.cat([dapi_img[:2],
#                              lamin_img[0].view(1,
#                                                lamin_img.shape[1],
#                                                lamin_img.shape[2])], dim=0)
        return tensor_img, img_path

    def __len__(self):
        return len(self.img_paths)


def get_celltest(root_path, use_channel='dapi'):
    dataset = TestDetection(root_path,
                            use_channel=use_channel)
    return dataset
