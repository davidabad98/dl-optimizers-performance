import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_kmnist_dataloaders(batch_size=64, num_workers=2):
    """
    Downloads the KMNIST dataset and returns DataLoaders for training and testing.

    Args:
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transformations: Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to mean 0.5, std 0.5
        ]
    )

    # Load training and testing datasets
    train_dataset = torchvision.datasets.KMNIST(
        root="../data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.KMNIST(
        root="../data", train=False, transform=transform, download=True
    )

    # Create DataLoaders for batch processing
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# Quick test
if __name__ == "__main__":
    train_loader, test_loader = get_kmnist_dataloaders()
    print(
        f"Train batch count: {len(train_loader)}, Test batch count: {len(test_loader)}"
    )
