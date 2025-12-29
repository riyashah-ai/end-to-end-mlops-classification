from torchvision import datasets , transforms
from torch.utils.data import DataLoader
from pathlib import Path

# data_path = "./data/raw"
# batch_size = 32
def get_data(data_path , batch_size):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])


    train_dataset = datasets.CIFAR10(
        root = data_path,
        train=True,
        download=False,
        transform=train_transform
    )


    test_dataset = datasets.CIFAR10(
        root = data_path,
        train=False,
        download=False,
        transform=test_transform
    )


    train_dataloader = DataLoader(train_dataset , batch_size=batch_size , shuffle = True)
    test_dataloader = DataLoader(test_dataset , batch_size=batch_size , shuffle = False)

    return train_dataloader , test_dataloader


# train_dataloader , test_dataloader = get_data(data_path , batch_size)
# for image , label in train_dataloader:

#     print(image.shape , label.shape)
# print(len(images))
