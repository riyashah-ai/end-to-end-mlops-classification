from torchvision import datasets , transforms
from pathlib import Path

data_dir = Path("./data/raw")
data_dir.mkdir(parents=True , exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

datasets.CIFAR10(
    root = data_dir,
    train = True,
    download=True,
    transform=transform
)
datasets.CIFAR10(
    root = data_dir,
    train = False,
    download=True,
    transform=transform
)

print("CIFAR10 dataset is downloaded sucessfully ")