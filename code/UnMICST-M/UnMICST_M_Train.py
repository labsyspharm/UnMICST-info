import datetime
import os
import time
import torch
import sys
sys.path.insert(0, 'ImageScience')
from mask_rcnn import maskrcnn_resnet50_fpn
from coco_utils import get_cellsegm
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_dataset(data_name, process_phase, transform_func, root_path,
                use_channel, augmentation='none', train=True,
                testdomain='clean'):
    path_list = {
        "cellsegm": (root_path, get_cellsegm, 2),
    }
    root_path, dataset_func, num_classes = path_list[data_name]

    ret_dataset = dataset_func(root_path, process_phase,
                               transform_func=transform_func,
                               use_channel=use_channel,
                               augmentation=augmentation,
                               train=train,
                               testdomain=testdomain)
    return ret_dataset, num_classes


def get_transform(is_train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    dataset_train, num_classes = get_dataset(args.dataset,
                                             'train',
                                             get_transform(is_train=True),
                                             args.root_path,
                                             use_channel=args.use_channel,
                                             augmentation=args.augmentation,
                                             train=True)
    dataset_valid, _ = get_dataset(args.dataset,
                                   'valid',
                                   get_transform(is_train=False),
                                   args.root_path,
                                   use_channel=args.use_channel,
                                   augmentation='none',
                                   train=False,
                                   testdomain=args.testdomain)
    dataset_test, _ = get_dataset(args.dataset,
                                  'test',
                                  get_transform(is_train=False),
                                  args.root_path,
                                  use_channel=args.use_channel,
                                  augmentation='none',
                                  train=False,
                                  testdomain=args.testdomain)
    print("Creating data loaders")
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        sampler_valid = torch.utils.data.distributed.DistributedSampler(
            dataset_valid)
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_valid = torch.utils.data.SequentialSampler(dataset_valid)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    batchsampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=batchsampler_train,
        num_workers=args.workers, collate_fn=utils.collate_fn)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1,
        sampler=sampler_valid, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=sampler_test, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = maskrcnn_resnet50_fpn(num_classes=num_classes,
                                  pretrained=args.pretrained)
    model.roi_heads.nms_thresh = 0.3
    model.roi_heads.detections_per_img = 1000
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        vis_dir = 'results/{}/{}'.format(args.resume.split('/')[-2],
                                         args.testdomain)
        evaluate(model, dataloader_test, device, is_test=True,
                 is_vis=args.vis, draw_bbox=False, vis_dir=vis_dir)
        return

    print("Start training")
    start_time = time.time()
    best_score = 0
    iter_count = 0
    warmup_factor = 1. / 1000
    warmup_iters = 1000
    warmup_scheduler = utils.warmup_lr_scheduler(
        optimizer, warmup_iters, warmup_factor)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        iter_count, _ = train_one_epoch(model, optimizer, warmup_scheduler,
                                        dataloader_train, device, epoch,
                                        iter_count, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            mAP_scores = evaluate(model, dataloader_valid, device=device)
            if best_score < mAP_scores['segm']:
                best_score = mAP_scores['segm']
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch},
                    os.path.join(args.output_dir, 'best_model.pth'))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--root-path', help='path to the dataset')
    parser.add_argument('--dataset', default='cellsegm', help='dataset')
    parser.add_argument(
        '--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='batch size per gpu (batch_size x num gpu)')
    parser.add_argument('--epochs', default=24, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22],
                        nargs='+', type=int,
                        help='decrease lr at epochs def: [16, 22]')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20,
                        type=int, help='print frequency')
    parser.add_argument('--output-dir', default='trained_models',
                        help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='start epoch')
    parser.add_argument("--test-only", dest="test_only",
                        help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained",
                        help="Use pre-trained models from the modelzoo",
                        action="store_true")
    parser.add_argument("--vis", help="Save visualization result",
                        action="store_true")
    parser.add_argument('--use-channel', default='dapi',
                        help='which channel to use; one of dapi/lamin/both')
    parser.add_argument('--augmentation', default='none', type=str,
                        help='types of augmentations <none/real/fake/ddll>')
    parser.add_argument('--testdomain', default='clean', type=str,
                        help='test channels <clean/topblur/bottomblur>')
    parser.add_argument('--rotation', help='apply rotation augmentation',
                        action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
