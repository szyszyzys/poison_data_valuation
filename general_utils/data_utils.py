import torch
from torch.utils.data import TensorDataset


def list_to_tensor_dataset(data: list[tuple[torch.Tensor, int]]) -> TensorDataset:
    # Separate images and labels from the list of tuples.
    images = torch.stack([img for img, label in data])
    labels = torch.tensor([label for img, label in data])
    return TensorDataset(images, labels)