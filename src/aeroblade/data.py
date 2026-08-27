from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision.transforms.v2 as tf
from PIL import Image
from torchvision.datasets import VisionDataset


def crop_to_multiple_of_8(img: Image.Image) -> Image.Image:
    """Center-crop an image so both dimensions are divisible by eight."""
    width, height = img.size
    new_width = width - (width % 8)
    new_height = height - (height % 8)

    if new_width == 0 or new_height == 0:
        raise ValueError(
            f"Image dimensions must be at least 8x8 pixels; got {width}x{height}."
        )
    if (new_width, new_height) == (width, height):
        return img

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return img.crop((left, top, left + new_width, top + new_height))


IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
DEFAULT_TRANSFORM = tf.Compose(
    [
        crop_to_multiple_of_8,
        tf.ToImage(),
        tf.ToDtype(torch.float32, scale=True),
    ]
)


class ImageFolder(VisionDataset):
    """
    Dataset for reading images from a list of paths, directories, or a mixture of both.
    """

    def __init__(
        self,
        paths: Union[list[Path], Path],
        transform: Optional[Callable] = DEFAULT_TRANSFORM,
        amount: Optional[int] = None,
    ) -> None:
        self.paths = [paths] if isinstance(paths, Path) else paths
        self.transform = transform
        self.amount = amount

        self.img_paths = []
        for path in self.paths:
            if path.is_dir():
                for file in read_files(path):
                    if file.suffix.lower() in IMG_EXTENSIONS:
                        self.img_paths.append(file)
                        if (
                            self.amount is not None
                            and len(self.img_paths) == self.amount
                        ):
                            break
            else:
                self.img_paths.append(path)

        if self.amount is not None and len(self.img_paths) < self.amount:
            raise ValueError("Number of images is less than 'amount'.")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[str, float]]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, str(self.img_paths[idx])

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Paths: {self.paths}")
        body.append(f"Transform: {repr(self.transform)}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


def read_files(path: Path) -> list[Path]:
    return sorted(path.iterdir())
