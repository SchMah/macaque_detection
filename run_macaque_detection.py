import Config
from ultralytics import YOLO
import os
import sys

def run_model(image = None):
    """
    Run the model on one or some images
    
    Input:
    "image" can take one of these forms: 
    None : an example image will be loaded
    str : a path to one single image
    list: list of multiple images
    Output : 
    """
    # if user does not specify an image, just load one example image from the config file
    if image is None:
        img_path = [Config.EXAMPLE_TEST_IMAGE]
    elif isinstance(image, str):
        img_path = [image]
    else:
        img_path = image
        
    
    # load the model
    model = YOLO(Config.BEST_MODEL_WEIGHTS)

    # run the model on a list of images
    results = model(img_path,
                   save=True,
                   project=Config.PROJECT_NAME,
                   name=Config.FOLDER_NAME,
                   exist_ok=True)

    
    # Process results list
    for im, result in zip(img_path,results):
        
        boxes = result.boxes  # Boxes object for bounding box outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen

        # save the image 
        #img_name = os.path.basename(im)
        #name, extension = os.path.splitext(img_name)

        # add time to the output image 
        #time_stamp = datetime.now().strftime("%Y%m%d-%H%M")
        # new name 
        #new_outpu_name = f"{name}_{time_stamp}{extension}"
        # output path
        #output_path = os.path.join(out_dir, new_outpu_name)
        
        #result.save(filename= output_path)  # save to disk
    
    
    return results


if __name__ == "__main__":
    img_arg = sys.argv[1:]
    if len(img_arg) == 0: # no image? then just run the sample image from the config file
        run_model()
    elif len(img_arg) == 1:
        run_model(image = img_arg[0])
    else:
        run_model(image = img_arg)

    