from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

DATASET_DIR = "/workspace/raid/data/dkorzh/datasets/"
PATH_TO_SAVE_MODELS = "/workspace/raid/data/dkorzh/models"


def imshow(inp, title=None):
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class EncodedDataset(Dataset):
    def __init__(self, dataset, encoder):
        self.dataset = dataset
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder.eval()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.encoder(image.unsqueeze(0)).squeeze(0)
        return image, label


def preprocess_fn(encoder, batch):
    images, labels = zip(*batch)
    with torch.no_grad():
        processed_batch = [
            encoder(image.unsqueeze(0)).squeeze(0) for image in images
            ]
    return torch.stack(processed_batch), labels


def prepare_data(batch_size=256, num_workers=4):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5))]
    )
    train_dataset = datasets.CIFAR10(
        root=DATASET_DIR, train=True, transform=transform, download=True
    )
    train_set, val_set = random_split(train_dataset, [45000, 5000])

    test_set = datasets.CIFAR10(
        root=DATASET_DIR, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def prepare_clf_data(
        train_set, val_set, test_set, enc,
        batch_size=256, num_workers=4
        ):
    train_set_encoded = EncodedDataset(train_set, enc)
    val_set_encoded = EncodedDataset(val_set, enc)
    test_set_encoded = EncodedDataset(test_set, enc)

    train_loader_encoded = DataLoader(
        train_set_encoded,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader_encoded = DataLoader(
        val_set_encoded,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader_encoded = DataLoader(
        test_set_encoded,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return (
        train_set_encoded,
        val_set_encoded,
        test_set_encoded,
        train_loader_encoded,
        val_loader_encoded,
        test_loader_encoded,
    )
