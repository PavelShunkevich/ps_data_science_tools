import torch
from torch.utils.data import DataLoader


def compute_mean_and_std(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation for RGB images in a dataset.

    Args:
        loader (DataLoader): DataLoader for iterating through the dataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and standard
                                       deviation tensors for RGB channels.
    """
    mean = torch.zeros(3)  # three channels (RGB)
    std = torch.zeros(3)  # three channels (RGB)
    total_images = 0  # for counting

    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)  # reshape to have 3 channels
        mean += torch.sum(images, dim=(0, 2))
        total_images += batch_size

    mean /= (total_images * 32 * 32)

    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        std += torch.sum((images - mean.view(1, 3, 1)) ** 2, dim=(0, 2))
    std /= (total_images * 32 * 32)
    std = torch.sqrt(std)

    return mean, std
