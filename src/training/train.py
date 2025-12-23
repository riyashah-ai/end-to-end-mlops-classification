import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
import mlflow
import mlflow.pytorch
from src.data.dataloader import get_data
from src.models.CNN import CNN_classifier
from src.utils.config import load_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Devide :",DEVICE)


def train_model(config_path : str):

    print(config_path)

    cfg = load_config(config_path)

    train_config = cfg["training"]
    # data_config = cfg["data"]
    mlflow_config = cfg["mlflow"]
    model_config = cfg["model"]

     # 🔹 SageMaker paths (fallback for local run)
    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "data/raw")
    model_dir = os.environ.get("SM_MODEL_DIR", "/home/riya/Desktop/end-to-end-mlops-classification/Model_save")

    train_dataloader , test_dataloader = get_data(data_dir , train_config["BATCH_SIZE"])

    model = CNN_classifier(num_classes=model_config["num_classes"]).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters() , lr = train_config["LR"])

    mlflow.set_experiment(mlflow_config["experinments_name"])

    with mlflow.start_run():

        mlflow.log_param("epochs" , train_config["EPOCH"])
        mlflow.log_param("Learning rate" , train_config["LR"])
        mlflow.log_param("Batch_size " , train_config["BATCH_SIZE"])
        mlflow.log_param("Model type" , model_config["name"])


        for epoch in range(train_config["EPOCH"]):
            
            # ============Training========== 
            model.train()
            running_loss = 0.0
            train_correct = 0 
            train_total = 0

            for images , labels in train_dataloader:

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs , labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Accuracy 

                train_total += labels.size(0)
                _ , train_pred = torch.max(outputs , 1)
                train_correct += (train_pred == labels ).sum().item() 
            
            avg_train_loss = running_loss / len(train_dataloader)
            avg_train_accuracy = (train_correct/train_total)*100


            #  ========== Validation ===========

            model.eval()

            val_total = 0 
            val_correct = 0
            val_loss = 0.0

            with torch.no_grad():

                for images , labels in test_dataloader:

                    images , labels = images.to(DEVICE), labels.to(DEVICE)

                    val_output = model(images)

                    loss = criterion(val_output , labels)

                    val_loss += loss.item()

                    _ , val_pred = torch.max(val_output , 1)
                    val_total += labels.size(0)
                    val_correct+= (val_pred == labels).sum().item()

                avg_val_loss = val_loss / len(test_dataloader)
                avg_val_accuracy = (val_correct/val_total)*100

            mlflow.log_metric("training_loss" , avg_train_loss , step=epoch)
            mlflow.log_metric("training_accuracy" , avg_train_accuracy , step=epoch)
            mlflow.log_metric("Validation_loss" , avg_val_loss , step=epoch)
            mlflow.log_metric("Validation_accuracy" , avg_val_accuracy , step=epoch)

            # 🔹 Save model to SageMaker output directory
            model_path = os.path.join(model_dir, "model.pth")
            torch.save(model.state_dict(), model_path)

            print(
                f"Epoch [{epoch+1}/{train_config['EPOCH']}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.2f}%"
            )

        mlflow.pytorch.log_model(model , artifact_path = "cnn_model" , registered_model_name="cifar10_classifier")

    return model



if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()
    if args.config:
        train_model(args.config)
    else:
        train_model()