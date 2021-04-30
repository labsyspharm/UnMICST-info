import copy
import math
import random
import torch

from PIL import Image

from torchvision.transforms import functional as F
# from torch.nn import functional as nnF

import pdb


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
                target["bg_maps"] = target["bg_maps"].flip(-1)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-2)
                target["bg_maps"] = target["bg_maps"].flip(-2)
        return image, target


class RandomRotation(object):
    def __call__(self, image, target):
        # k_num = random.randint(0, 4)
        k_num = random.randint(0, 3)
        height, width = image.shape[-2:]
        image = image.rot90(k_num, [-2, -1])
        bbox = target["boxes"]
        if k_num == 1:
            x_minmax = bbox[:, [1, 3]]
            y_minmax = width - bbox[:, [2, 0]]
            bbox[:, [0, 2]] = x_minmax
            bbox[:, [1, 3]] = y_minmax
        elif k_num == 2:
            x_minmax = width - bbox[:, [2, 0]]
            y_minmax = height - bbox[:, [3, 1]]
            bbox[:, [0, 2]] = x_minmax
            bbox[:, [1, 3]] = y_minmax
        elif k_num == 3:
            y_minmax = bbox[:, [0, 2]]
            x_minmax = height - bbox[:, [3, 1]]
            bbox[:, [0, 2]] = x_minmax
            bbox[:, [1, 3]] = y_minmax
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = target["masks"].rot90(k_num, [-2, -1])
            target["bg_maps"] = target["bg_maps"].rot90(k_num, [-2, -1])
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 # size=(800, 800),
                 scale=(0.2, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        # if isinstance(size, (tuple, list)):
        #     self.size = size
        # else:
        #     self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, image, target):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        while 1:
            trans_target = copy.deepcopy(target)
            y, x, h, w = self.get_params(image, self.scale, self.ratio)
            (width, height) = image.size

            tran_image = F.crop(image, y, x, h, w)
            # image = F.resized_crop(image, y, x, h, w,
            #                        self.size, self.interpolation)
            bboxes = trans_target["boxes"]  # l, t, r, b
            bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], y, y+h) - y
            bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], x, x+w) - x
            bool_list = ((bboxes[:, 3] - bboxes[:, 1]) !=
                         0) & ((bboxes[:, 2] - bboxes[:, 0]) != 0)
            if bool_list.sum() == 0:
                continue
            trans_target["boxes"] = bboxes[bool_list]
            if "masks" in trans_target:
                trans_target["masks"] = \
                    trans_target["masks"][bool_list, y:y+h, x:x+w]
                trans_target["bg_maps"] = \
                    trans_target["bg_maps"][bool_list, y:y+h, x:x+w]
                trans_target["labels"] = trans_target["labels"][bool_list]
                trans_target["area"] = trans_target["area"][bool_list]
                trans_target["iscrowd"] = trans_target["iscrowd"][bool_list]
            break
        return tran_image, trans_target

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + \
            '('  # size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4)
                                                    for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4)
                                                    for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
