import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import random

from torch.utils import data
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from tifffile import imsave

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

def test_model(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best saved model
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Initalize dataloader 
    data_loader = get_loader(args.dataset)
    data_path = "/n/pfister_lab2/Lab/vcg_biology/cycif/DapiUnetTrainingData/LPTCGSdapilaminRTAug64/"
    loader = data_loader(
        data_path,
        split="test",
    )
    
    n_classes = loader.n_classes

    test_loader = data.DataLoader(
        loader,
        batch_size=1, # for testing purposes
        num_workers=8,
    )

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Directory to save Probability Maps and Segmentation Map
    run_id = random.randint(1, 100000)
    out_path = os.path.join("runs/psp_segmenter_tests", str(run_id))
    os.makedirs(out_path,exist_ok=True)
    print("OUT_PATH: {}".format(out_path))

    # Test the model
    with torch.no_grad():
        for i, (images, label) in tqdm(enumerate(test_loader)):
            image_name_list = list(images.keys())
            image_list = list(images.values())

            N_channels = len(image_list)
            N_channels = (N_channels - 1) if N_channels > 1 else 1
            
            for j in range(N_channels): #(6) test model on 6 DAPI channels for each test image
                image_name = image_name_list[j]
                image_orig = image_list[j]

                # Extract Lamin channel for visualization
                if (N_channels-1 == 0):
                    image_lamin_orig = image_list[N_channels-1] # extract NES channel
                else:
                    image_lamin_orig = image_list[N_channels]
                image_lamin = image_lamin_orig.to(device)

                images = image_orig.to(device)
                label = label.to(device)

                outputs = model(images)

                # Resize output of network to same size as label
                target_size = (label.size()[1],label.size()[2])
                outputs = torch.nn.functional.interpolate(outputs,size=target_size,mode='bicubic')
                
                outputs = torch.nn.functional.softmax(outputs,dim=1)

                #Downsample original images to target size for visualization
                images = torch.nn.functional.interpolate(images,size=target_size,mode='bicubic')
                #Downsample lamin images to target size for visualization
                image_lamin = torch.nn.functional.interpolate(image_lamin,size=target_size,mode='bicubic')

                contours = (outputs[:,1,:,:]).unsqueeze(dim=1)
                nuclei = (outputs[:,2,:,:]).unsqueeze(dim=1)

                # Extract target label for nuclei and contour classes
                np_label = label.cpu().numpy()

                nuclei_target = np.zeros_like(np_label,np.uint8)
                nuclei_target[np_label == 2] = 1
                nuclei_target = torch.from_numpy(nuclei_target)
                nuclei_target = nuclei_target.type(torch.float).to(device)

                contour_target = np.zeros_like(np_label,np.uint8)
                contour_target[np_label == 1] = 1
                contour_target = torch.from_numpy(contour_target)
                contour_target = contour_target.type(torch.float).to(device)

                # Probability maps visualization below
                img = []
                nuc = []

                img_lamin = []
                con = []

                # Normalize individual images in batch
                bs = 0 # batch size = 1 
                img.append( (images[bs][0] - images[bs][0].min()) / (images[bs][0].max() - images[bs][0].min()) )
                nuc.append( nuclei[bs] )  
                
                img_lamin.append( (image_lamin[bs][0] - image_lamin[bs][0].min()) / (image_lamin[bs][0].max() - image_lamin[bs][0].min()) )                
                con.append( contours[bs] ) 
                
                img = [torch.unsqueeze(elem,dim=0) for elem in img] #expand dim=0 for images in batch
                img_lamin = [torch.unsqueeze(elem,dim=0) for elem in img_lamin] #expand dim=0 for images in batch
                
                # Convert normalized batch to Tensor
                img_tensor = torch.cat((img),dim=0)
                nuc_tensor = torch.cat((nuc),dim=0)

                img_lamin_tensor = torch.cat((img_lamin),dim=0)
                con_tensor = torch.cat((con),dim=0)

                # Nuclei Output
                nuc_output = torch.cat((img_tensor, nuc_tensor, nuclei_target), dim=0)
                nuc_output = nuc_output.unsqueeze(dim=1)
                save_image(make_grid(nuc_output, nrow=3), os.path.join(out_path, image_name + "_Nuc.png"), normalize=False)

                # Contour Output
                con_output = torch.cat((img_lamin_tensor, con_tensor, contour_target), dim=0)
                con_output = con_output.unsqueeze(dim=1)
                save_image(make_grid(con_output, nrow=3), os.path.join(out_path, image_name + "_Con.png"), normalize=False)

                # # Actual segmentation map prediction
                # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                # decoded = loader.decode_segmap(pred)
                # print("Classes found: ", np.unique(pred))
                # misc.imsave(os.path.join(out_path, "Seg_map_%d.tif" % i), decoded)
                # #save_image(make_grid(decoded,nrow=1),os.path.join(out_path, "Seg_map_%d.tif" % i), normalize=False)
                # print("Segmentation Mask Saved at: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="DNA_NoAug",
        help="Dataset to use",
    )
    
    args = parser.parse_args()
    test_model(args)