import argparse
import sys
import numpy as np
import mlflow
from PIL import Image
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Image classification using MLflow model")
    parser.add_argument("--model_uri",type=str,required=True,help="MLflow model URI (e.g. models:/my_model/Production or runs:/<run_id>/model)")
    parser.add_argument("--image_path",type=str,required=True,help="Path to input image")
    parser.add_argument(
        "--image_size",type=int,default=32,help="Image size expected by the model (default: 224)")
    return parser.parse_args()

def load_and_preprocess_image(image_path, image_size):
    """
    Load image, resize, normalize, and convert to model input format
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    image = image.resize((image_size, image_size))
    image = np.array(image).astype("float32") / 255.0
    image = np.permute_dims(image, (2,0,1))

    # Add batch dimension: (1, H, W, C)
    image = np.expand_dims(image, axis=0)
    image = torch.Tensor(image)
    # print(type(image))
    # exit()
    return image



def main():
    args = parse_args()

    CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
    # Load MLflow model
    print(f"Loading model from: {args.model_uri}")
    model = mlflow.pytorch.load_model(args.model_uri)

    # Preprocess image
    image = load_and_preprocess_image(args.image_path, args.image_size)

    # Run inference
    predictions = model(image)

    _ , train_pred = torch.max(predictions , 1)
    predicted_cls = CLASSES[train_pred]
    

    print("Prediction result")
    print("-----------------")
    print(f"Image path      : {args.image_path}")
    print(f"Predicted class : {predicted_cls}")
    # if "confidence" in locals():
    #     print(f"Confidence      : {confidence:.4f}")



if __name__ == "__main__":
    main()