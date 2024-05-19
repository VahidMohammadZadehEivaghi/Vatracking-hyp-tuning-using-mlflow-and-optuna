from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import random_split


def load_data(batch_size):
    train_set = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    val_set = FashionMNIST(
        root="data",
        download=True,
        train=False,
        transform=transforms.ToTensor()
    )

    train_subset, test_subset = random_split(
        train_set, [0.8, 0.2]
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data(16)
    for x, y in train_loader:
        print(x.shape, y.shape)