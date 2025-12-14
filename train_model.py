from ultralytics import YOLO
import pandas as pd
import os
import cv2
import Config
import data_prep


def data_readiness():
    """
    Check whether the dataset exists before running the model. if not, create a new one. 
    If data_prep.py was executed previously, the folder Config.DATA_PATH ( 'Macaque_datasets') should exist in the parent folder. This function performs a sanity check.
    """
    if not os.path.isdir(Config.DATA_PATH):
        print("Data folder not found. Creating one...")
        all_images = data_prep.concat_images()
        data_prep.make_ready_train_dir() 
        data_prep.split_train_val_test(all_images)
        data_prep.apply_yolo_format_to_all()
    else: 
        print("Dataset is already created")


def model_training():
    """
    Initialize and train the YOLO model based on parameters specified in the configuration file. 
    """
    #Initialize
    print(f"Loading base model {Config.BASE_MODEL}")
    model = YOLO(Config.BASE_MODEL)

    # Train
    print("Start training...")
    results = model.train(data = Config.MODEL_DATA,
                      epochs = Config.EPOCH,
                      batch = Config.BATCH,
                      device = Config.DEVICE,
                      optimizer = Config.OPTIMIZER,
                      imgsz = Config.IMAGE_SIZE,
                      augment = Config.AUGMENT,
                      patience = Config.PATIENCE)
    return model, results



if __name__ == "__main__":
    data_readiness()
    trained_model, training_results = model_training()
    
    
    
    