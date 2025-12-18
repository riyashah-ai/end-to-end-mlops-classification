import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from src.data.dataloader import get_data
from src.models.CNN import CNN_classifier
import mlflow
import mlflow.pytorch

EPOCH = 20
BATCH_SIZE = 32
lr = 0.001
DEVICE = "cpu"
DATA_PATH = "./data/raw"

def train_model(epoch , batch_size , lr ,device = DEVICE):

    train_dataloader , test_dataloader = get_data(DATA_PATH , BATCH_SIZE)

    model = CNN_classifier().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters() , lr = lr)

    mlflow.set_experiment("CIFAR_Classifier_dvc")

    with mlflow.start_run():

        mlflow.log_param("epochs" , EPOCH)
        mlflow.log_param("Learning rate" , lr)
        mlflow.log_param("Batch_size " , BATCH_SIZE)
        mlflow.log_param("Model type" , "CNN_classifier")


        for epoch in range(EPOCH):

            model.train()
            running_loss = 0.0

            for images , labels in train_dataloader:

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs , labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_dataloader)

            mlflow.log_metric("training_loss" , avg_loss , step=epoch)

            print(f"Epoch : {epoch+1} , loss : {avg_loss:.4f}")

        mlflow.pytorch.log_model(model , artifact_path = "cnn_model")

    return model



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(epoch=EPOCH , lr= lr , batch_size=BATCH_SIZE , device=device)

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved as model.pth")