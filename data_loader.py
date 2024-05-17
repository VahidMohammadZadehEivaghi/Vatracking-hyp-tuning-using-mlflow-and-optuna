from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def load_data_and_transformation(batch_size):
    transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]
    )

    training_set = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transformation
    )
    validation_set = FashionMNIST(
        root="data",
        download=True,
        train=False,
        transform=transformation
    )

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, transformation


if __name__ == "__main__":
    tr_loader, v_loader, tr = load_data_and_transformation(16)
    for x, y in tr_loader:
        print(x.shape, y.shape)