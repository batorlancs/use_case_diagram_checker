import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import math

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
        # minimum padding to ensure the shape is not cut off when rotated
        self.img_min_padding = ((math.sqrt(2) * img_size) - img_size) / 2
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
        res = torch.where(res > 0, 1, 0).type(torch.uint8)
        return res

    def resize_image_random(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Resizes the image by a random factor.
        """
        if self.img_size < 200:
            random_size = self.rng.integers(
                low=self.img_size/4, high=self.img_size/2)
        else:
            random_size = self.rng.integers(
                low=self.img_size/8, high=self.img_size/4)

        resizer = v2.Resize(
            size=(int(random_size), int(random_size)), antialias=True)
        res = resizer(image)
        return res, random_size

    def rectangle(self):
        """
        Returns a binary image of a rectangle. (uint8)
        """
        rec_shape_size = self.img_size
        minimum_gap = self.img_size/8

       # generate two points that connect the rectangle and make sure they are at least 1/3 of the image size apart
        while True:
            a, b = self.rng.integers(
                low=self.img_min_padding, high=self.img_size-self.img_min_padding, size=2)
            c, d = self.rng.integers(low=self.img_min_padding, high=self.img_size-self.img_min_padding, size=2)
            vx = c - a
            vy = d - b
            if abs(vx) > minimum_gap and abs(vy) > minimum_gap:
                break

        # generate the coordinates of the rectangle
        xx, yy = rectangle_perimeter(start=(a, b), end=(
            c, d), shape=(rec_shape_size, rec_shape_size))
        rectangle = xx, yy

        img = self.get_empty_image()
        img[rectangle] = 1
        img = self.rotate_image_random(img.unsqueeze(0))
        return img

    def line(self):
        """
        Returns a binary image of a line. (uint8)
        """
        # Choose two points that are at least 1/3 of the image size apart
        while True:
            a, b = self.rng.integers(low=0, high=self.img_size, size=2)
            c, d = self.rng.integers(low=0, high=self.img_size, size=2)
            vx = c - a
            vy = d - b
            norm = np.sqrt(vx**2 + vy**2)
            if norm > self.img_size/12 and norm < self.img_size/3:
                break

        xx, yy, _ = line_aa(a, b, c, d)
        line = xx, yy

        img = self.get_empty_image()
        img[line] = 1
        return img.unsqueeze(0)

    def ellipse(self):
        """
        Returns a binary image of an ellipse. (uint8)
        """
        min_radius = self.img_size/12
        max_radius = self.img_size/4

        # Random radius
        r_radius = self.rng.integers(
            low=min_radius, high=max_radius)
        
        # generate c_radius that is at least half as big or smaller as r_radius
        c_radius = self.rng.integers(
            low=r_radius//4, high=r_radius//2)

        # Random centre coordinate
        r, c = self.rng.integers(
            low=self.img_min_padding, high=self.img_size-self.img_min_padding, size=2)

        xx, yy = ellipse_perimeter(
            r, c, r_radius, c_radius, shape=(self.img_size, self.img_size))
        ellipse = xx, yy

        img = self.get_empty_image()
        img[ellipse] = 1
        img = self.rotate_image_random(img.unsqueeze(0))
        return img.type(torch.uint8)

    def stickman(self):
        """
        Returns a binary image of a stickman. (uint8)
        """
        stickman = self.get_random_stickman()
        stickman = self.rotate_image_random(stickman)
        stickman, stickman_size = self.resize_image_random(stickman)
        img = self.get_empty_image()

        # get random coordinates
        a, b = self.rng.integers(
            low=0, high=self.img_size-stickman_size, size=2)

        # pad the image with zeros
        padding = (a, b, self.img_size-a-stickman_size,
                   self.img_size-b-stickman_size)
        padder = v2.Pad(padding=padding, fill=0, padding_mode='constant')
        img = padder(stickman)
        return img.type(torch.uint8)

    def dashed_arrow(self):
        """
        Returns a binary image of a dashed arrow. (uint8)
        """
        # Choose two start and end coordinates where the norm is between the range
        while True:
            a, b = self.rng.integers(low=0, high=self.img_size, size=2)
            c, d = self.rng.integers(low=0, high=self.img_size, size=2)
            vx = c - a
            vy = d - b
            norm = np.sqrt(vx**2 + vy**2)
            if norm > self.img_size/12 and norm < self.img_size/3:
                break

        GAP = self.rng.integers(low=norm/16, high=norm/8)
        DASH = self.rng.integers(low=norm/8, high=norm/4)

        xx, yy, _ = line_aa(a, b, c, d)

        points_to_remove = []
        for i in range(len(xx)):
            if i % (GAP + DASH) >= DASH:
                points_to_remove.append(i)

        xx = np.delete(xx, points_to_remove)
        yy = np.delete(yy, points_to_remove)

        line = xx, yy
        img = self.get_empty_image()
        img[line] = 1

        # normalize
        vx = vx / norm
        vy = vy / norm
        perpendicular_vx = -vy
        perpendicular_vy = vx

        # get point on the line random pixels from the start
        distance = float(self.rng.integers(low=int(norm/6), high=int(norm/3)))

        x, y = a + vx * distance, b + vy * distance

        # get two points on the perpendicular line
        perpendicular_distance = self.rng.integers(
            low=4, high=max(8, int(norm/4)))
        x1, y1 = x + perpendicular_vx * perpendicular_distance, y + \
            perpendicular_vy * perpendicular_distance
        x2, y2 = x - perpendicular_vx * perpendicular_distance, y - \
            perpendicular_vy * perpendicular_distance

        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.img_size - 1))
        y1 = max(0, min(y1, self.img_size - 1))
        x2 = max(0, min(x2, self.img_size - 1))
        y2 = max(0, min(y2, self.img_size - 1))

        xx1, yy1, _ = line_aa(int(x1), int(y1), int(a), int(b))
        line1 = xx1, yy1
        img[line1] = 1

        xx2, yy2, _ = line_aa(int(x2), int(y2), int(a), int(b))
        line2 = xx2, yy2
        img[line2] = 1

        return img.unsqueeze(0)
