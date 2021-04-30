import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from tifffile import imsave
from functools import reduce

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from torch.utils.tensorboard import SummaryWriter

def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Setup Augmentations
    # augmentations = cfg["training"].get("augmentations", None)
    # data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    v_loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        drop_last=True,
    )

    valloader = data.DataLoader(
        v_loader, 
        batch_size=cfg["training"]["batch_size"], 
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        drop_last=True,
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model_orig = get_model(cfg["model"], n_classes).to(device)
    if cfg["training"]["pretrain"] == True:
        # Load a pretrained model
        model_orig.load_pretrained_model(
            model_path="pretrained/pspnet101_cityscapes.caffemodel"
        )
        logger.info("Loaded pretrained model.")
    else:
        # No pretrained model
        logger.info("No pretraining.")

    model = torch.nn.DataParallel(model_orig, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    ### Visualize model training

    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
    def matplotlib_imshow(data, is_image):
        if is_image: #for images
            data = data / 4 + 0.5     # unnormalize
            npimg = data.numpy()
            plt.imshow(npimg, cmap="gray")
        else: # for labels
            nplbl = data.numpy()
            plt.imshow(t_loader.decode_segmap(nplbl))

    def plot_classes_preds(data, batch_size, iter, is_image=True):
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(1, batch_size, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(data[idx], is_image)

            ax.set_title("Iteration Number "+str(iter))

        return fig

    best_iou = -100.0
    #best_val_loss = -100.0
    i = start_iter
    flag = True
    
    #Check if params trainable
    print('CHECK PARAMETER TRAINING:')
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(name, param.data)

    while i <= cfg["training"]["train_iters"] and flag:
        for (images_orig, labels_orig, weights_orig, nuc_weights_orig) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train() #convert model into training mode
            images = images_orig.to(device)
            labels = labels_orig.to(device)
            weights = weights_orig.to(device)  
            nuc_weights = nuc_weights_orig.to(device)          

            optimizer.zero_grad()

            outputs = model(images)

            # Transform output to calculate meaningful loss
            out = outputs[0]

            # Resize output of network to same size as labels
            target_size = (labels.size()[1],labels.size()[2])
            out = torch.nn.functional.interpolate(out,size=target_size,mode='bicubic')

            # Multiply weights by loss output
            loss = loss_fn(input=out, target=labels)

            loss = torch.mul(loss,weights) # add contour weights
            loss = torch.mul(loss,nuc_weights) # add nuclei weights
            loss = loss.mean() # average over all pixels to obtain scaler for loss

            loss.backward() # computes gradients over network
            optimizer.step() #updates parameters

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0: # frequency with which visualize training update
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )
                #Show mini-batches during training
                
                # #Visualize only DAPI
                # writer.add_figure('Inputs',
                #     plot_classes_preds(images_orig.squeeze(), cfg["training"]["batch_size"], i, True),
                #             global_step=i)

                # writer.add_figure('Targets',
                #     plot_classes_preds(labels_orig, cfg["training"]["batch_size"], i, False),
                #             global_step=i)

                #Take max across classes (of probability maps) and assign class label to visualize semantic map
                #1)
                out_orig = torch.nn.functional.softmax(outputs[0],dim=1).max(1).indices.cpu()
                #out_orig = out_orig.cpu().detach()
                #2) 
                #out_orig = torch.argmax(outputs[0],dim=1)
                #3)
                #out_orig = outputs[0].data.max(1)[1].cpu()

                # #Visualize predictions 
                # writer.add_figure('Predictions',
                #     plot_classes_preds(out_orig, cfg["training"]["batch_size"], i, False),
                #             global_step=i)
                
                #Save probability map
                prob_maps_folder = os.path.join(writer.file_writer.get_logdir(),"probability_maps")
                os.makedirs(prob_maps_folder,exist_ok=True)

                #Downsample original images to target size for visualization
                images = torch.nn.functional.interpolate(images,size=target_size,mode='bicubic')

                out = torch.nn.functional.softmax(out,dim=1)

                contours = (out[:,1,:,:]).unsqueeze(dim=1)
                nuclei = (out[:,2,:,:]).unsqueeze(dim=1)
                background = (out[:,0,:,:]).unsqueeze(dim=1)

                #imageTensor = torch.cat((images, contours, nuclei, background),dim=0)
                               
                # Save images side by side: nrow is how many images per row
                #save_image(make_grid(imageTensor, nrow=2), os.path.join(prob_maps_folder,"Prob_maps_%d.tif" % i))

                # Targets visualization below
                nplbl = labels_orig.numpy()
                targets = [] #each element is RGB target label in batch
                for bs in np.arange(cfg["training"]["batch_size"]):
                    target_bs = t_loader.decode_segmap(nplbl[bs])
                    target_bs = 255*target_bs
                    target_bs = target_bs.astype('uint8')
                    target_bs = torch.from_numpy(target_bs)
                    target_bs = target_bs.unsqueeze(dim=0)
                    targets.append(target_bs) #uint8 labels, shape (N,N,3)
                
                target = reduce(lambda x,y: torch.cat((x,y), dim = 0), targets)
                target = target.permute(0,3,1,2) # size=(Batch, Channels, N, N)
                target = target.type(torch.FloatTensor)

                save_image(make_grid(target, nrow=cfg["training"]["batch_size"]), os.path.join(prob_maps_folder,"Target_labels_%d.tif" % i))
                
                
                # Weights visualization below:
                #wgts = weights_orig.type(torch.FloatTensor)
                #save_image(make_grid(wgts, nrow=2), os.path.join(prob_maps_folder,"Weights_%d.tif" % i))

                # Probability maps visualization below
                t1 = []
                t2 = []
                t3 = []
                t4 = []
                
                # Normalize individual images in batch
                for bs in np.arange(cfg["training"]["batch_size"]):
                    t1.append( (images[bs][0] - images[bs][0].min()) / (images[bs][0].max() - images[bs][0].min()) )
                    t2.append( contours[bs] ) 
                    t3.append( nuclei[bs] ) 
                    t4.append( background[bs] ) 
                
                t1 = [torch.unsqueeze(elem,dim=0) for elem in t1] #expand dim=0 for images in batch
                # Convert normalized batch to Tensor
                tensor1 = torch.cat((t1),dim=0)
                tensor2 = torch.cat((t2),dim=0)
                tensor3 = torch.cat((t3),dim=0)
                tensor4 = torch.cat((t4),dim=0)

                tTensor = torch.cat((tensor1, tensor2, tensor3, tensor4),dim=0)
                tTensor = tTensor.unsqueeze(dim=1)

                save_image(make_grid(tTensor, nrow=cfg["training"]["batch_size"]), os.path.join(prob_maps_folder,"Prob_maps_%d.tif" % i), normalize=False)

                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1) # adds value to history (title, loss, iter index)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]: # evaluate model on validation set at these intervals
                model.eval() # evaluate mode for model
                with torch.no_grad():
                    for i_val, (images_val, labels_val, weights_val, nuc_weights_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        weights_val = weights_val.to(device)
                        nuc_weights_val = nuc_weights_val.to(device)

                        outputs_val = model(images_val)

                        # Resize output of network to same size as labels
                        target_val_size = (labels_val.size()[1],labels_val.size()[2])
                        outputs_val = torch.nn.functional.interpolate(outputs_val,size=target_val_size,mode='bicubic')

                        # Multiply weights by loss output 
                        val_loss = loss_fn(input=outputs_val, target=labels_val)

                        val_loss = torch.mul(val_loss,weights_val)
                        val_loss = torch.mul(val_loss,nuc_weights_val)
                        val_loss = val_loss.mean() # average over all pixels to obtain scaler for loss

                        outputs_val = torch.nn.functional.softmax(outputs_val,dim=1)

                        #Save probability map
                        val_prob_maps_folder = os.path.join(writer.file_writer.get_logdir(),"val_probability_maps")
                        os.makedirs(val_prob_maps_folder,exist_ok=True)

                        #Downsample original images to target size for visualization
                        images_val = torch.nn.functional.interpolate(images_val,size=target_val_size,mode='bicubic')

                        contours_val = (outputs_val[:,1,:,:]).unsqueeze(dim=1)
                        nuclei_val = (outputs_val[:,2,:,:]).unsqueeze(dim=1)
                        background_val = (outputs_val[:,0,:,:]).unsqueeze(dim=1)

                        # Targets visualization below
                        nplbl_val = labels_val.cpu().numpy()
                        targets_val = [] #each element is RGB target label in batch
                        for bs in np.arange(cfg["training"]["batch_size"]):
                            target_bs = v_loader.decode_segmap(nplbl_val[bs])
                            target_bs = 255*target_bs
                            target_bs = target_bs.astype('uint8')
                            target_bs = torch.from_numpy(target_bs)
                            target_bs = target_bs.unsqueeze(dim=0)
                            targets_val.append(target_bs) #uint8 labels, shape (N,N,3)
                        
                        target_val = reduce(lambda x,y: torch.cat((x,y), dim = 0), targets_val)
                        target_val = target_val.permute(0,3,1,2) # size=(Batch, Channels, N, N)
                        target_val = target_val.type(torch.FloatTensor)

                        save_image(make_grid(target_val, nrow=cfg["training"]["batch_size"]), os.path.join(val_prob_maps_folder,"Target_labels_%d_val_%d.tif" % (i,i_val)))
                        
                        # Weights visualization below:
                        #wgts_val = weights_val.type(torch.FloatTensor)
                        #save_image(make_grid(wgts_val, nrow=2), os.path.join(val_prob_maps_folder,"Weights_val_%d.tif" % i_val))

                        # Probability maps visualization below
                        t1_val = []
                        t2_val = []
                        t3_val = []
                        t4_val = []
                        # Normalize individual images in batch
                        for bs in np.arange(cfg["training"]["batch_size"]):
                            t1_val.append( (images_val[bs][0] - images_val[bs][0].min()) / (images_val[bs][0].max() - images_val[bs][0].min()) )
                            t2_val.append( contours_val[bs] ) 
                            t3_val.append( nuclei_val[bs] ) 
                            t4_val.append( background_val[bs] )

                        t1_val = [torch.unsqueeze(elem,dim=0) for elem in t1_val] #expand dim=0 for images_val in batch
                        # Convert normalized batch to Tensor
                        tensor1_val = torch.cat((t1_val),dim=0)
                        tensor2_val = torch.cat((t2_val),dim=0)
                        tensor3_val = torch.cat((t3_val),dim=0)
                        tensor4_val = torch.cat((t4_val),dim=0)

                        tTensor_val = torch.cat((tensor1_val, tensor2_val, tensor3_val, tensor4_val),dim=0)
                        tTensor_val = tTensor_val.unsqueeze(dim=1)

                        save_image(make_grid(tTensor_val, nrow=cfg["training"]["batch_size"]), os.path.join(val_prob_maps_folder,"Prob_maps_%d_val_%d.tif" % (i,i_val)), normalize=False)

                        pred = outputs_val.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                ### Save best validation loss model
                # if val_loss_meter.avg >= best_val_loss:
                #     best_val_loss = val_loss_meter.avg
                #     state = {
                #         "epoch": i + 1,
                #         "model_state": model.state_dict(),
                #         "optimizer_state": optimizer.state_dict(),
                #         "scheduler_state": scheduler.state_dict(),
                #         "best_val_loss": best_val_loss,
                #     }
                #     save_path = os.path.join(
                #         writer.file_writer.get_logdir(),
                #         "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                #     )
                #     torch.save(state, save_path)
                ###

                score, class_iou = running_metrics_val.get_scores() # best model chosen via IoU
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                
                ### Save best mean IoU model
                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
                ###

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/psp_segmenter.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    print("IMAGE SHAPE: {}".format(cfg["data"]["img_rows"]))
    print("BATCH SIZE: {}".format(cfg["training"]["batch_size"]))
    print("LEARNING RATE: {}".format(cfg["training"]["optimizer"]["lr"]))

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Here we go!")

    train(cfg, writer, logger)
