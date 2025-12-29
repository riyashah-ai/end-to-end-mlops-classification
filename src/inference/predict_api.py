import io
import mlflow
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image

MODEL_URI = "models:/cifar10_classifier/1"

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model ONCE at startup
model = mlflow.pytorch.load_model(MODEL_URI)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    print(image.shape)
    with torch.no_grad():
        outputs = model(image)
        _ , pred = torch.max(outputs , 1)

        print(CLASSES[pred])
        print(pred)

    return {
        "prediction": CLASSES[pred],
        "class_id": pred
    }
