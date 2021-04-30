from vision import VisionDataset
from PIL import Image, ImageFilter
import os
import os.path
import tifffile
import numpy as np
from skimage import filters
import pdb
import cv2


class CocoCaptions(VisionDataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoCaptions, self).__init__(
            root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, use_channel='dapi', augmentation='none', train=True, testdomain='clean'):
        super(CocoDetection, self).__init__(
            root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.use_channel = use_channel
        self.augmentation = augmentation
        self.train = train
        self.gaussian_sigmas = [0, 0.5, 0.75, 1]
        self.testdomain = testdomain
        self.ddll_dict = {
            'd1l1': 0, 'd1l2': 1, 'd3l1': 2, 'd3l2': 3, 'd6l1': 4, 'd6l2': 5}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        # path = 'I00034_Img.tif'

        bg_map = coco.loadImgs(img_id)[0]['bg_map']
        for t in target:
            t['bg_map'] = bg_map
        # np_img[0-1]: DAPI clean
        # np_img[2-3]: DAPI saturated
        # np_img[4-5]: DAPI blurry
        # np_img[6:]: Lamin clean
        np_img = tifffile.imread(os.path.join(self.root, path))
        if self.use_channel == 'dapi':
            if self.train:
                if self.augmentation == 'real':
                    rand_id = np.random.randint(6)
                    img = Image.fromarray(np_img[rand_id]).convert('RGB')
                elif self.augmentation == 'realblur':
                    rand_id = 2*np.random.randint(3)
                    img = Image.fromarray(np_img[rand_id]).convert('RGB')
                else:
                    if self.augmentation == 'fake':
                        if np.random.rand() < .5:
                            sigma_id = np.random.randint(
                                len(self.gaussian_sigmas))
                            gauss_sigma = self.gaussian_sigmas[sigma_id]
                            filtered_img = filters.gaussian(np_img[0]/255.,
                                                            gauss_sigma)
                            np_img[0] = (filtered_img*255).astype(np.uint8)
                    elif self.augmentation != 'none':
                        # learned ddll augmentation
                        if np.random.rand() < .5:
                            ddll_id = self.ddll_dict[self.augmentation]
                            aug_path = '{}_{}.png'.format(
                                path.split('.tif')[0], ddll_id)
                            np_img = cv2.imread(
                                os.path.join(self.root, aug_path), 0)
                            np_img = np.expand_dims(np_img, 0)
                    img = Image.fromarray(np_img[0]).convert('RGB')
            else:
                if self.testdomain == 'clean':
                    img = Image.fromarray(np_img[0]).convert('RGB')
                elif self.testdomain == 'topblur':
                    img = Image.fromarray(np_img[2]).convert('RGB')
                elif self.testdomain == 'bottomblur':
                    img = Image.fromarray(np_img[4]).convert('RGB')
                else:
                    raise Exception('invalid input for --testdomain.')
        elif self.use_channel == 'lamin':
            img = Image.fromarray(np_img[6]).convert('RGB')
        elif self.use_channel == 'both':
            if self.train:
                if self.augmentation == 'real':
                    rand_id = np.random.randint(6)
                    dapi_img = Image.fromarray(np_img[rand_id]).convert('RGB')
                else:
                    if (self.augmentation == 'fake') & (np.random.rand() < .5):
                        sigma_id = np.random.randint(len(self.gaussian_sigmas))
                        gauss_sigma = self.gaussian_sigmas[sigma_id]
                        filtered_img = filters.gaussian(np_img[0]/255.,
                                                        gauss_sigma)
                        np_img[0] = (filtered_img*255).astype(np.uint8)
                    dapi_img = Image.fromarray(np_img[0]).convert('RGB')
            else:
                if self.testdomain == 'clean':
                    dapi_img = Image.fromarray(np_img[0]).convert('RGB')
                elif self.testdomain == 'topblur':
                    dapi_img = Image.fromarray(np_img[2]).convert('RGB')
                elif self.testdomain == 'bottomblur':
                    dapi_img = Image.fromarray(np_img[4]).convert('RGB')
            lamin_img = Image.fromarray(np_img[6]).convert('RGB')
            img = [dapi_img, lamin_img]
        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # if type(pil_img) == list:
        #     dapi_img, _ = self.transforms(pil_img[0], target)
        #     lamin_img, target = self.transforms(pil_img[1], target)
        #     # First two: DAPI, Third: Lamin
        #     img = torch.cat([dapi_img[:2], lamin_img[0]], dim=0)
        # else:
        #     img, target = self.transforms(pil_img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
