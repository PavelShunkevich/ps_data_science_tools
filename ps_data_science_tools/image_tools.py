import matplotlib.pyplot as plt
import numpy as np

def show_color_rgb(rgb_values: np.array):

    # Создаем массив 1x1x3, представляющий один пиксель
    pixel_color = rgb_values.reshape(1, 1, 3)

    # Отображаем пиксель
    plt.imshow(pixel_color)
    plt.title("Color from RGB values")
    plt.show()

def show_picture_cifar10(tuple_image: tuple):
    """
    Params:
        tuple_image: image tensor, label (int)
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    class_names_dict = dict(zip(range(len(class_names)), class_names))

    img, label = tuple_image
    print(class_names_dict[label])
    plt.imshow(img.permute(1, 2, 0))
    plt.show()