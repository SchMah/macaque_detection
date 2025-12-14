"""
This script collects all images/labels from raw directory (MacaqueImagePairs/), splits them to train/val/test, and store them in separate folders (by default: Macaque_datasets/)

Macaque_datasets/
│
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images 
│
├── labels/
│   ├── train/      # labels for training images
│   └── val/        # labels for validation images
│   └── test/       # labels for test images   


"""

# import libraries 
import random
import os
import Config
import shutil
import pandas as pd

random.seed(Config.RANDOM_SEED)

image_dir = Config.RAW_MACAQUE_DATASETS # downloaded images
root_path = Config.DATA_PATH 

"""
1) Go through folders (0_100 through 9_100) and concatenate all images and labels from all sub folders. 
"""
def concat_images():
    all_images = []
    missing_labels = 0
    # Choose only folders of interest (there are irrelevant files in the directory)
    sub_folders = Config.SUB_FOLDERS
    
    for folder in sub_folders: 
        folder_images = os.path.join(image_dir, folder, "images")
        folder_labels = os.path.join(image_dir, folder, "labels_with_ids")
        
        for filename in os.listdir(folder_images):
            if filename.endswith(".jpg"):
                single_image_path = os.path.join(folder_images, filename)
                single_label = os.path.splitext(filename)[0]+".txt"
                single_label_path = os.path.join(folder_labels, single_label)
                if os.path.exists(single_label_path):
                    all_images.append((single_image_path, single_label_path))
                else:
                    missing_labels += 1
                    

    print(f"There are {len(all_images)} images in the parent directory {image_dir} ")
    print(f"There are {missing_labels} images with no corresponding label file")
    return all_images




def make_ready_train_dir():
    """
    This function checks whether final folders for training the model exists, if not create the folders. 
    Two folders (images and labels) and each folder with three subfolders(train/val/test) are needed. 
    """
    
    folders = ["images/train", "images/val", "images/test", "labels/train", "labels/val","labels/test"] 
    # I added this part to prevent overwriting images. Alternatively, I could assign a prefix(e.g., date) to create a new dataset
    if os.path.isdir(root_path):
        shutil.rmtree(root_path)
        print(f"Removing existing folder {root_path}")
           
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
        print(f"Creating {root_path} folder")

    for folder in folders:
        if not os.path.isdir(os.path.join(root_path, folder)):
            os.makedirs(os.path.join(root_path, folder))
            print(f"Creating sub folder : {os.path.join(root_path,folder)}")
        else:
            print(f"folder {os.path.join(root_path,folder)} exists")




def split_train_val_test(all_images, train_ratio = Config.TRAIN_RATIO, val_ratio= Config.VAL_RATIO):
    """
    This function splits all images randomly into three training/validation/test datasets and copy the images and corresponding labels to folders created in make_ready_train_dir() function.
    """
    random.shuffle(all_images)
    total_n = len(all_images)
    n_train = int(train_ratio * total_n)
    n_val = int(val_ratio * total_n)
    n_test = total_n - n_train - n_val

    train_data = all_images[:n_train]  
    val_data = all_images[n_train:n_train +n_val]
    test_data = all_images[n_train+n_val:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    def copy_files(data_sample, dest_folder):
        for file in range(len(data_sample)):
            source_image = data_sample[file][0] # the first one is the image
            source_label = data_sample[file][1] # the second one is the label
            
            shutil.copy(source_image, os.path.join(root_path,"images",dest_folder ))
            
            shutil.copy(source_label, os.path.join(root_path,"labels",dest_folder ))

    copy_files(train_data, "train")
    copy_files(val_data, "val")
    copy_files(test_data, "test")



def remove_id_column(label_dir):
    """
    Convert label files to YOLO format.
    Currently labels are saved as : 
    <class_id> <running_counter_of_objects> <x_center> <y_center> <width> <height>
    The accepted format is: 
    <class_id> <x_center> <y_center> <width> <height>
    
    """
    
    for filename in os.listdir(label_dir):
        label_path = os.path.join(label_dir, filename)
        df = pd.read_csv(label_path, sep = " ", header = None)
        if df.shape[1] == 6:
            df.columns = ["class_id", "running_counter_of_objects", "x_center", "y_center", "width", "height"]
            df = df[["class_id", "x_center", "y_center", "width", "height"]]
            # Write again in YOLO format
            df.to_csv(label_path, sep=" ", header=False, index=False)

def apply_yolo_format_to_all():
    for folder in ["train", "val", "test"]:
        label_dir = os.path.join(root_path, "labels", folder)
        if os.path.isdir(label_dir):
            print(f"Processing label files in {label_dir}")
            remove_id_column(label_dir)
       

if __name__ == "__main__":
    all_images = concat_images()
    make_ready_train_dir()
    split_train_val_test(all_images)
    apply_yolo_format_to_all()
    
