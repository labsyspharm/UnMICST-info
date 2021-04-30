# UnMICST-M - (Mask R-CNN)
## Requirements

- Linux
- Python 3.6
- PyTorch 1.5.1
- CUDA 10.1

## Installation Instructions

    conda create --name unmicst python=3.6
    conda activate unmicst
    pip install -r requirements.txt

## Operation Instructions
### Dataset

Training data can be downloaded from `/training data` from https://www.dropbox.com/sh/3aqp83f5w1pxk0y/AABFgNRMJD2EvfSLFgCrXrBba?dl=0<br>
The dataset is supposed to be arranged below
  
```
RootFolder
├── test  
│   ├── *_Img.tif
├── train
│   ├── *_Img.tif
├── valid
│   ├── *_Img.tif
├── coco_cellsegm_test.json
├── coco_cellsegm_train.json
├── coco_cellsegm_valid.json
```
    

### Train
- Set `nproc_per_node` and `world-size` as the number of GPUs to use
- `root-path` is a path to a folder that contains train / val / test data
- `output-dir` is a path to save trained models

*DNA Channel / Real Augmentation*
    
    ./DNA_Aug.sh

*DNA Channel / Gaussian Augmentation*
    
    ./DNA_GaussianAug.sh

*DNA Channel / No Augmentation*
    
    ./DNA_NoAug.sh

*DNA + NES Channels / Real Augmentation*
    
    ./DNA_NES_Aug.sh

*DNA + NES Channels / No Augmentation*
    
    ./DNA_NES_NoAug.sh


### Test
- Set `nproc_per_node` and `world-size` as the number of GPUs to use
- `use-channel` is either `dapi`/`both`
- `testdomain` is one of  `clean`/`topblur`/`bottomblur`
- `resume` is a path to a saved model to test
- `root-path` is a path to a folder that contains train / val / test data
- `output-dir` is a path to save trained models

*Command for Testing*

    ./UnMICST_M_Test.sh
