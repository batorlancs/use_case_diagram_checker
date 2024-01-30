# from new_draw_class import Draw
from utils import show
import numpy as np
import torch
from skimage.draw import line_aa, rectangle_perimeter, ellipse_perimeter
from torchvision.transforms import v2


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

        return None

    def load_stickman_images(self) -> list:
        """
        Loads the stickman images from the stickman folder.
        """
        pass

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

    def rectangle_outline(self):
        """
        Returns the image coordinates of a rectangle outline
        """
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
        """
        Returns the image coordinates of a line.
        """
        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(low=0, high=self.img_size, size=2)
        c, d = self.rng.integers(low=0, high=self.img_size, size=2)

        # print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")
        xx, yy, _ = line_aa(a, b, c, d)

        line = xx, yy

        return line

    def ellipse(self):
        """
        Returns the image coordinates of an ellipse.
        """
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
        """
        Returns the image coordinates of a stickman
        """



img_size = 1000
rng = np.random.default_rng()

draw = Draw(img_size, rng)

test = draw.ellipse()
show(test)
