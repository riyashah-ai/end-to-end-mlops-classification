import os
import random
import mlflow
import mlflow.pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataloader import get_data
from src.utils.config import load_config
# ----------------------------
# CONFIG
# ----------------------------
config = load_config("/home/riya/Desktop/end-to-end-mlops-classification/configs/train.yaml")

MODEL_URI =  "models:/cifar10_classifier/1"
OUTPUT_DIR = "artifacts/predictions"
NUM_IMAGES = 20
Data_path = config["data"]["data_path"]
BATCH_SIZE = config["training"]["BATCH_SIZE"]


    # data_config = cfg["data"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CIFAR-10 labels
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# ----------------------------
# DATASET
# ----------------------------
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# test_dataset = torchvision.datasets.CIFAR10(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform
# )

# test_loader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=64,
#     shuffle=False
# )


_ , test_dataloader = get_data(Data_path, BATCH_SIZE)
# ----------------------------
# LOAD MODEL
# ----------------------------
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# EVALUATION
# ----------------------------
correct = 0
total = 0

all_images = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        print(type(images))
        exit()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_images.extend(images.cpu())
        all_preds.extend(predicted.cpu())
        all_labels.extend(labels.cpu())

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# ----------------------------
# SAVE RANDOM PREDICTIONS
# ----------------------------
indices = random.sample(range(len(all_images)), NUM_IMAGES)

for i, idx in enumerate(indices):
    img = all_images[idx]
    pred = all_preds[idx].item()
    label = all_labels[idx].item()

    # Denormalize
    # img = img * 0.5 + 0.5
    np_img = img.permute(1, 2, 0).numpy()

    plt.figure()
    plt.imshow(np_img)
    plt.title(f"GT: {CLASSES[label]} | Pred: {CLASSES[pred]}")
    plt.axis("off")

    file_path = os.path.join(OUTPUT_DIR, f"prediction_{i}.png")
    plt.savefig(file_path)
    plt.close()

# ----------------------------
# LOG TO MLFLOW
# ----------------------------
with mlflow.start_run(run_name="cifar10_evaluation"):
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="predictions")

print("Evaluation completed and artifacts logged to MLflow")
