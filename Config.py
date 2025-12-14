"""
Conf file
"""
from datetime import datetime
import torch
# -----------------------
# ------------ General Data Directories -------------
# -----------------------

# Path to raw data (downloaded from https://data.goettingen-research-online.de/file.xhtml?persistentId=doi:10.25625/CMQY0Q/ZH2YKH&version=1.0)
RAW_MACAQUE_DATASETS = "MacaqueImagePairs"  

# Find relevant folders :10 folders, 0_100 through 9_100
SUB_FOLDERS = []
for i in range(10):
    SUB_FOLDERS.append(f"{i}_100")

# Path to data folder where images/labels are splitted for model training. This folder will be created automatically as part of data preparation. No need to create it manually. In case, a different name is preferred, give it another name

DATA_PATH = 'Macaque_datasets'

RANDOM_SEED = 42

# -----------------------
# ------------ Model Training Config -------------
# -----------------------
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# path to yolo dataset config file. 
MODEL_DATA = "data.yaml"

# base yolo model to start training from. 
BASE_MODEL = "yolo11m.pt"

# device to run training on. Options are 'mps' (if using mac), cuda, or cpu
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


# Hyperparameters
EPOCH = 200
BATCH = 16
IMAGE_SIZE = 640
AUGMENT = True
PATIENCE = 50
OPTIMIZER = 'auto'

# -----------------------
# ------------ Model Outputs -------------
# -----------------------

# where the model is saved:
BEST_MODEL_WEIGHTS = "runs/detect/train/weights/best.pt"

# An example image
EXAMPLE_TEST_IMAGE = "demo/img_00372_1.jpg"

# in case fo visualizing some images, save the output images in a new folder
PROJECT_NAME = 'macaque_detection_runs'
FOLDER_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")