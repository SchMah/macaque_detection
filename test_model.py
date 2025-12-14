from ultralytics import YOLO
import Config

def model_eval():
    """
    Loads the best trained model weights and runs validation on the test dataset.
    This provides performance metrics (e.g., mAP, Precision, Recall)
    The output is also in runs/detect/val
    """
    print("Loading the best model weights")
    model = YOLO(Config.BEST_MODEL_WEIGHTS)
    
    print("Evaluating model on test data")
    model_eval_metrics = model.val(data = "data.yaml", split = 'test')
    print("Model performance on test dataset")
    print(f"Mean average precision at IoU 0.5: {model_eval_metrics.box.map50: 0.3f}")
    print(f"Precision: {model_eval_metrics.box.mp: 0.3f}")
    print(f"Recall: {model_eval_metrics.box.mr: 0.3f}")
    print(f"F1 score: {model_eval_metrics.box.f1[0]: 0.3f}")
    return model_eval_metrics


if __name__ == "__main__":
    model_eval()