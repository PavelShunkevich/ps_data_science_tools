import matplotlib.pyplot as plt
import torch
import numpy as np

def show_color_rgb(rgb_values: tuple[float, float, float]) -> None:
    """
    Displays a color patch based on the provided RGB values.

    Args:
        rgb_values (Tuple[float, float, float]): A tuple containing three floats representing
        the red, green, and blue components of the color, respectively.
        Each float value should be between 0 and 1.

    Returns:
        None
    """

    # Convert the tuple to a NumPy array
    rgb_array: np.ndarray = np.array(rgb_values)

    # Reshape to a 1x1x3 array for displaying as a single pixel color
    pixel_color: np.ndarray = rgb_array.reshape(1, 1, 3)

    # Display the pixel
    plt.imshow(pixel_color)
    plt.title("Color from RGB values")
    plt.show()

def show_picture_cifar10(image_tuple: tuple[torch.Tensor, np.ndarray]) -> None:
    """
    Displays a CIFAR-10 image along with its class label.

    Args:
        image_tuple (Tuple[Union[torch.Tensor, np.ndarray], int]): A tuple containing:
            - The image data as either a PyTorch tensor or a NumPy array.
              The image data should have the shape (C, H, W) where C is
              the number of channels (e.g., 3 for RGB), H is height and W is
              the width.
            - The integer class label associated with the image.

    Returns:
        None
    """
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    class_names_dict = dict(zip(range(len(class_names)), class_names))

    image, label = image_tuple

    if isinstance(image, torch.Tensor):
         image = image.permute(1, 2, 0).cpu().numpy() # Move to cpu before converting to numpy
    elif not isinstance(image, np.ndarray):
        raise TypeError(
            "Input image must be either a PyTorch Tensor or a NumPy array."
        )


    print(f"Class: {class_names_dict[label]}")
    plt.imshow(image)
    plt.show()