# from new_draw_class import Draw
from utils import show
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from skimage.draw import line_aa, rectangle_perimeter, ellipse_perimeter
from torchvision.transforms import v2
from typing import Tuple


class Draw:
    """
    Class used to draw shapes onto images. Methods return coordinates of 
    corresponding shape on a 2d np array of shape (img_size, img_size). 
    The np rng is used for enabling derministic behaviour. 

    Args:
    img_size (int): draws onto 2d array of shape (img_size, img_size).
    rng (Generator): used for enabling deterministic behaviour. Example 
        of valid rng: rng = np.random.default_rng(12345)

    """

    def __init__(self, img_size: int, rng: np.random.Generator) -> None:
        self.img_size = img_size
        self.rng = rng
        self.stickman_dataset = self.load_stickman_images()
        return None

    def load_stickman_images(self) -> torchvision.datasets.ImageFolder:
        """
        Loads the stickman images from the stickman folder.
        """
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1-x),
        ])

        dataset_folder = "dataset"
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_folder, transform=transform)
        return dataset

    def get_random_stickman(self) -> torch.Tensor:
        """
        Returns a random stickman image.
        """
        idx = self.rng.integers(low=0, high=len(self.stickman_dataset))
        return self.stickman_dataset[idx][0]

    def get_empty_image(self) -> torch.Tensor:
        """
        Returns an empty image of size (img_size, img_size)
        """
        return torch.zeros(size=(self.img_size, self.img_size), dtype=torch.uint8)

    def rotate_image_random(self, image: torch.Tensor) -> torch.Tensor:
        """
        Rotates the image by a random angle.
        """
        # Rotate the shape.
        rotater = v2.RandomRotation(degrees=(0, 180))
        res = rotater(image)
        return res

    def resize_image_random(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Resizes the image by a random factor.
        """
        random_size = self.rng.integers(
            low=self.img_size/8, high=self.img_size/4)
        resizer = v2.Resize(
            size=(int(random_size), int(random_size)), antialias=True)
        res = resizer(image)
        return res, random_size

    def rectangle_outline(self):
        # generate start and end points for rectangle
        a, b = self.rng.integers(low=0, high=self.img_size, size=2)
        c, d = self.rng.integers(low=0, high=self.img_size, size=2)

        # print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")

        # generate the coordinates of the rectangle
        xx, yy = rectangle_perimeter(start=(a, b), end=(
            c, d), shape=(self.img_size, self.img_size))
        rectangle_outline = xx, yy

        img = self.get_empty_image()
        img[rectangle_outline] = 1
        img = self.rotate_image_random(img.unsqueeze(0))
        return img

    def line(self):
        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(low=0, high=self.img_size, size=2)
        c, d = self.rng.integers(low=0, high=self.img_size, size=2)

        # print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")
        xx, yy, _ = line_aa(a, b, c, d)

        line = xx, yy

        return line

    def ellipse(self):
        # Random centre coordinate
        r, c = self.rng.integers(
            low=self.img_size/3, high=self.img_size/1.5, size=2)

        # Random radius
        r_radius, c_radius = self.rng.integers(
            low=self.img_size/20, high=self.img_size/4, size=2)

        # print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")
        xx, yy = ellipse_perimeter(
            r, c, r_radius, c_radius, shape=(self.img_size, self.img_size))
        ellipse = xx, yy

        img = self.get_empty_image()
        img[ellipse] = 1
        img = self.rotate_image_random(img.unsqueeze(0))
        return img

    def stickman(self):
        stickman = self.get_random_stickman()
        stickman = self.rotate_image_random(stickman)
        stickman, stickman_size = self.resize_image_random(stickman)
        img = self.get_empty_image()

        print("img", img.shape)
        print("stickman", stickman.shape, stickman_size)

        # get random coordinates
        a, b = self.rng.integers(
            low=0, high=self.img_size-stickman_size, size=2)

        # pad the image with zeros
        padding=(a, b, self.img_size-a-stickman_size,
                        self.img_size-b-stickman_size)
        padder = v2.Pad(padding=padding, fill=0, padding_mode='constant')
        img = padder(stickman)
        return img
    
    def dashed_arrow(self):
        GAP = 10
        DASH = 30

        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(low=0, high=self.img_size, size=2)
        c, d = self.rng.integers(low=0, high=self.img_size, size=2)

        # print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")
        xx, yy, _ = line_aa(a, b, c, d)

        points_to_remove = []
        # keep 20 points then remove 10 points
        for i in range(len(xx)):
            if i % (GAP + DASH) >= DASH:
                points_to_remove.append(i)

        xx = np.delete(xx, points_to_remove)
        yy = np.delete(yy, points_to_remove)

        print(xx)

        print("dashed arrow")
        print(xx, yy)
        print(xx.shape, yy.shape)

        line = xx, yy
        img = self.get_empty_image()
        img[line] = 1
        return img


img_size = 500
rng = np.random.default_rng()

draw = Draw(img_size, rng)

test = draw.ellipse()
show(test)

stickman = draw.stickman()
show(stickman)

arrow = draw.dashed_arrow()
show(arrow)

stickman.shape, stickman.dtype, stickman.max(), stickman.min()
