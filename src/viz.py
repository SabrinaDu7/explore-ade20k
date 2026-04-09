import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_sample(image: Image.Image, mask: Image.Image | None = None, title: str = ""):
    cols = 2 if mask is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")
    if mask is not None:
        axes[1].imshow(np.array(mask), cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def show_grid(images: list[Image.Image], titles: list[str] | None = None, cols: int = 4):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(titles[i] if titles else "")
        ax.axis("off")
    plt.tight_layout()
    return fig
