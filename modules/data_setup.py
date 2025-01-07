import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(dataset_name, batch_size=128, input_size=(224, 224), download=True):
    """
    Preparess data loaders for the specified dataset.
    Args:
        dataset_name (str): e.g., 'CIFAR-10', 'CIFAR-100', 'FashionMNIST', 'Food-101').
        batch_size (int): Batch size.
        input_size (tuple): Size to resize images to (default is 224x224 for ViT).
        download (bool): Whether to download the dataset if not already present.
    Returns:
        tuple: train_loader, test_loader
    """
    if dataset_name == "CIFAR-10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        dataset_class = datasets.CIFAR10
    elif dataset_name == "CIFAR-100":
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        dataset_class = datasets.CIFAR100
    elif dataset_name == "FashionMNIST":
        mean, std = (0.5,), (0.5,)
        dataset_class = datasets.FashionMNIST
    elif dataset_name == "Food-101":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        dataset_class = datasets.Food101
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Define transforms for training and testing
    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(input_size[0], padding=4) if dataset_name != "Food-101" else transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Loading datasets
    if dataset_name == "Food-101":
        train_dataset = dataset_class(root="./data", split="train", download=download, transform=transform_train)
        test_dataset = dataset_class(root="./data", split="test", download=download, transform=transform_test)
    else:
        train_dataset = dataset_class(root="./data", train=True, download=download, transform=transform_train)
        test_dataset = dataset_class(root="./data", train=False, download=download, transform=transform_test)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

