import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def load_kmnist(data_dir="../data"):
    """
    Loads the KMNIST dataset with transformations.

    Args:
        data_dir (str): Directory to store the dataset.

    Returns:
        torchvision.datasets.KMNIST: Loaded dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = torchvision.datasets.KMNIST(
        root=data_dir, train=True, transform=transform, download=True
    )
    return dataset


def get_kmnist_dataloaders(batch_size=64, num_workers=2, valid_split=0.1):
    """
    Downloads the KMNIST dataset and returns DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses for data loading.
        valid_split (float): Fraction of training data to use for validation.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Define transformations: Convert to tensor and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load full training dataset
    full_train_dataset = torchvision.datasets.KMNIST(
        root="../data", train=True, transform=transform, download=True
    )

    # Split training dataset into training and validation sets
    train_size = int((1 - valid_split) * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    train_dataset, valid_dataset = random_split(
        full_train_dataset, [train_size, valid_size]
    )

    # Load test dataset
    test_dataset = torchvision.datasets.KMNIST(
        root="../data", train=False, transform=transform, download=True
    )

    # Create DataLoaders for batch processing
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


# Quick test
if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_kmnist_dataloaders()
    print(
        f"Train batch count: {len(train_loader)}, Valid batch count: {len(valid_loader)}, Test batch count: {len(test_loader)}"
    )
