# UnMICST-U (UNet)

## Installation instructions
1. Install python 3.7 and Anaconda
2. Create a conda environment: `conda create -n tf_gpu tensorflow-gpu=1.12 cudatoolkit=9.0`
3. Activate conda environment : `conda activate tf_gpu`
4. Install the following packages: `conda install scikit-image` and `conda install -c conda-forge opencv`

## Operation instructions
1. Download Python scripts and `ImageScience` folder from `code/UnMICST-U` folder. 
2. Download models from dropbox `models/UnMicst-U`. https://www.dropbox.com/sh/3aqp83f5w1pxk0y/AABFgNRMJD2EvfSLFgCrXrBba?dl=0 Unzip and copy the **subfolder** to the same level as the script file(s).
4. Download the training data from the `training data` dropbox folder. Unzip the folder to the same level as the script file(s).

Run model on new images by `python <script name>.py train/test/deploy <append parameters below>`
1. `function` : whether to `train` a new model, `test` existing model on test data, or `deploy` the model on a completely new image.
2. `--imagePath` : absolute path to an image you wish to deploy model on.
3. `--outputPath` : specify where to save the output files after deploying model on new image.
4. `--channel` : specify the channel(s) to be used. For DNA only models, only one channel should be specified. For DNA_NES models, use 2 channels ie. `--channel 0 3`
5. `--scalingFactor` : an upsample or downsample factor if your pixel sizes are mismatched from the dataset.
