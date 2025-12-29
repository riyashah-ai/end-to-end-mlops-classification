from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path

# Base directory
data_dir = Path("./data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Output directory for test images
output_dir = data_dir / "cifar10_test_images"
output_dir.mkdir(parents=True, exist_ok=True)

# Transform (keep it simple)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load test dataset
test_dataset = datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

# CIFAR-10 class names
class_names = test_dataset.classes
print(class_names)

# Create class subfolders
for class_name in class_names:
    (output_dir / class_name).mkdir(parents=True, exist_ok=True)

# Save images
for idx, (image, label) in enumerate(test_dataset):
    class_name = class_names[label]
    image_path = output_dir / class_name / f"{idx}.png"
    save_image(image, image_path)

print("CIFAR-10 test images saved successfully!")
