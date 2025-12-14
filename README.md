# Macaque Localization

## Repository Structure
Note that some folders below (e.g., `MacaqueImagePairs`, `Macaque_datasets`) are created locally and not included in the GitHub repository. 
This is how the project is organized once everything is set up. 
 
```
├── MacaqueImagePairs # Raw downloaded dataset folders
│   ├── 0_100
│   └── 9_100
│   
├── Macaque_datasets 
│   ├── images # Contains train/val/test images
│   └── labels # Contains train/val/test labels
│
├── runs ## Output folder for YOLO training/validation results/logs
│   └── detect  
│
├── macaque_detection_runs ## Output folder for inference results
│   ├── 20251209_181554
│
├── Notebooks
│   └── Visualization_pre.ipynb # Initial data visualization
├── Config.py # configuration file
├── data.yaml # YOLO dataset config file
├── data_prep.py # Script to split data  
├── requirements.txt # List of Python dependencies
├── run_macaque_detection.py # Inference script for running prediction on new images 
├── test_model.py # Model evaluation on test set
├── train_model.py # Train the model 
├── yolo11m.pt # Base model
└── Readme.md
```

## Overview
This repository contains Python scripts to detect macaques in wild images. It used the YOLO (You Only Look Once) architecture to generate bounding boxes around detected macaques. For this project, I chose YOLO since the available macaque dataset is relatively small. So, starting from a pre-trained model, which was originally trained on a large dataset, is an appropriate transfer learning approach. This allows the model to reuse general low-level features (e.g., edges, shapes) and then fine-tune them to a single class (“Macaque”). I used the “m” variant to balance speed and accuracy. Additionally, in contrast to region-proposal methods, YOLO sees the whole image during training which helps reduce the background error (false positives).


## Installation
First, obtain the source code by cloning this repository using Git: 

```bash
 git clone https://github.com/SchMah/macaque_detection.git
```
Then, navigate to the project directory: 

```bash
cd macaque_detection
````

## Setting up the environment
I recommend using a virtual environment to isolate the project dependencies from your main Python installation. Alternatively, just make sure you use Python3+ and that the libraries listed in ***requirements.txt*** are installed. 

Requirements: Python3.11+ installed. 

### Option 1: (Using Conda) 

#### Requirements : Anaconda must be installed. 
#### Create and activate the Conda environment:
 ***macaque_localization*** is an arbitrary name. You may want to give it another name, if you prefer.
 
```bash 
# create a conda environment
conda create -n macaque_localization python=3.11 pip

# activate conda
conda activate macaque_localization
```
#### Install dependencies: 
Install the required packages:

```bash
pip install -r requirements.txt
```

### Option 2 (Using venv)
#### Requirements: Python 3.11 installed

#### Create and activate a virtual environment: 
 ***macaque_localization*** is an arbitrary name. But `venv` is also a common convention. 

```bash
# Create the environment 
Python3 -m venv macaque_localization
```

```bash
# Activate the virtual enviornmnet
source macaque_localization/bin/activate
```

#### Install dependencies: 
Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Acquisition:
To run the full training and evaluation pipeline, you must download the full dataset from [this link](https://data.goettingen-research-online.de/file.xhtml?persistentId=doi:10.25625/CMQY0Q/ZH2YKH&version=1.0!) and place it in the project folder. 
After downloading the directory structure should look like this :
```
├── MacaqueImagePairs # Raw downloaded dataset folders
│   ├── 0_100
│   └── 9_100
│   
```

The script `data_prep.py` will create `Macaque_datasets/` automatically with the correct `images/` and `labels/` subfolders.

## Usage

If you want to use the trained model in your analysis, use the following commands to put bounding boxes around new images.

### Using a default example image:
If you are curious to see the inference without providing exlicitly an image, simply execute the script without any arguments. The script uses a default example image specified in the configuration file (`Config.py`)

``` bash
python run_macaque_detection.py
```
### Using specified images:
To analyze your own images, provide the paths to your images after the script name. 
 
**For a single image**:
```bash
python run_macaque_detection.py "Macaque_datasets/images/test/img_00156_1.jpg"
```

**For multiple images**: 
Separate the file paths with spaces

```bash
python run_macaque_detection.py "Macaque_datasets/images/test/img_00156_1.jpg" "Macaque_datasets/images/test/img_00339_0.jpg" "Macaque_datasets/images/test/img_00397_1.jpg"

```
### Output
**Displaying the image**
When you run the script, a window will pop up. It displays the image with the bounding boxes around any detected macaques.

**Saving the results**
The output images are saved under the main directory in a folder named `macaque_detection_runs`. Results are automatically separated into subfolders specified by data and time (`"%Y%m%d_%H%M%S"`) of the run. For instance *macaque_detection_runs/20251209_181554*


## Training the Model
**Important:**
YOLO requires a `data.yaml` file to find the prepared data and number of classes to expect. Make sure it is in the root directory.
```yaml
path: Macaque_datasets
train: images/train
val: images/val
test: images/test

nc: 1
names: ['Macaque']
```
To train the model from scratch, execute the training script. 

```bash
python train_model.py 
```

### Steps included in `train_model.py`

1. **Data Preparation:** The script first checks if the preprocessed dataset exists. If not, it runs the `data_prep.py` pipeline to:
    - Concatenate images from all raw data subfolders (`0_100` to `0_900`) in `MacaqueImagePairs/` folder.
    - Split the data into Train(80%), Validation(10%), and Test(10%) sets.
    - Convert the raw label file provided as <class_id> <running_counter_of_objects> <x_center> <y_center> <width> <height> to compatible YOLO format <class_id> <x_center> <y_center> <width> <height>
    
2.  **Model Initialization:** it loads the pre-trained `yolo11m.pt` base model.

3.  **Training:** The model is fine-tune on `macaque_datasets` with the parameters specified in `Config.py`

4.  **Model Saving:**
    - The best model weights are automatically saved to `runs/detect/train/weights/best.pt`
    - Saving logs including metrics like loss and mAP over epochs are saved to `runs/detect/train`

Note that all hyperparameters (epochs, batch size, and etc.) can be adjusted in `Config.py`

## Evaluating the Model
To evaluate the performance of the model on the held-oout test set, run:

```
python test_model.py
```
This will load the best model during training (`best.pt`) and reports key performance metrics. YOLO saves the results and detailed metrics in the `run/detect/val` folder.  
