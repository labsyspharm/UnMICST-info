# UnMICST-P (PSPNet)

Pyramid Scene Parsing Network (PSPNet) Semantic Segmentation for CyCIF Images collected at Harvard Medical School. 

The project structure and baseline PSPNet model is adapted from: https://github.com/meetshah1995/pytorch-semseg.
Please refer to this repository for additional details. 

## Installation Instructions 
Create a virtual environment. 

Install all of the required packages using:

```pip install -r requirements.txt```

Edit the Configuration file (or add your own) in the ```configs/``` directory.
Let ```cfg``` be the Configuration file. Depending on your desired experiment, change the ```cfg['data']['dataset']``` value to one of 
```
DNA_Aug
DNA_GaussianAug
DNA_NES_Aug
DNA_NES_NoAug
DNA_NoAug
```

## Operation Instructions
Determine whether or not you would like to load a pretrained model. To load a pretrained model, place the model file in the ```pretrained/```
folder in https://www.dropbox.com/sh/3aqp83f5w1pxk0y/AABFgNRMJD2EvfSLFgCrXrBba?dl=0 and set the ```cfg["training"]["pretrained"] = True```.  Otherwise, set ```cfg["training"]["pretrained"] = False```.

1) To train the model, on a SLURM compute node, execute:

```sbatch train.py```

The ```slurm/``` folder will contain a standard output and standard error file associated with your experiment. 

2) To train the model in a conventional Python environment, execute: 

```python -u train.py --config configs/psp_segmenter.yml```

You may generate and use your own Configuration file, if desired.

The ```runs/``` directory will automatically populate with an experiment folder that has the same name as the Configuration file and 
contain a dedicated folder with a uniquely generated run_id.

To test a desired model after training is complete, the model path and the dataset will need to be specified. Models can be downloaded in the `/models/UnMicst-P` folder of 
https://www.dropbox.com/sh/3aqp83f5w1pxk0y/AABFgNRMJD2EvfSLFgCrXrBba?dl=0 <br>
Note, the test dataset must match the dataset type used for training. 

That is, training on 

```DNA_Aug, DNA_GaussianAug, DNA_NoAug``` requires testing using ```DNA_test``` 

and training on

```DNA_NES_Aug, DNA_NES_NoAug``` requires testing using ```DNA_NES_test```

For example, to test the model associated with experiment run 9999 on dataset without augmentations (e.g. DNA_test), execute:

```python -u test_pspnet.py --model_path "runs/psp_segmenter/9999/pspnet_segmenter_best_model.pkl" --dataset "DNA_test" ```

A SLURM sbatch script called ```test.sh``` has also been included for convenience.



