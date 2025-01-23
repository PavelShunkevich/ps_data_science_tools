import torch

def compute_mean_and_std(loader):
    """
    compute_mean_and_std for RGB images (cifar10)
    """
    mean = torch.zeros(3) # three channels (RGB)
    std = torch.zeros(3) # three channels (RGB)
    total_images = 0 # for counting

    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1) # reshape to have 3 channels. Initially 4 channels (last two height and width)
        mean += torch.sum(images, dim=(0, 2))
        total_images += batch_size

    mean /= (total_images * 32 * 32)

    for images, _ in loader:
       batch_size = images.size(0)
       images = images.view(batch_size, 3, -1)
       std += torch.sum((images - mean.view(1,3,1))**2, dim=(0, 2)) # use broadcastiong too
    std /= (total_images * 32 * 32)
    std = torch.sqrt(std)

    return mean, std