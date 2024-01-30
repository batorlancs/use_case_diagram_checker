import torch
import numpy as np
from skimage.draw import line_aa, circle_perimeter_aa, rectangle_perimeter
from torchvision.transforms.functional import rotate


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
    
    def get_empty_image(self) -> torch.Tensor:
        """
        Returns an empty image of size (img_size, img_size)
        """
        return torch.zeros((self.img_size, self.img_size))
    
    def rotate_image_random(self, image: torch.Tensor) -> torch.Tensor:
        """
        Rotates the image by a random angle.
        """
        # Randomly choose an angle.
        angle = self.rng.integers(0, 360)
        # Rotate the shape.
        image = rotate(image, angle)
        return image
    
    def rectangle_outline(self):
        """
        Returns the image coordinates of a rectangle outline
        """
        # generate start and end points for rectangle
        a, b = self.rng.integers(low=0, high=self.img_size, size=2)
        c, d = self.rng.integers(low=0, high=self.img_size, size=2)

        print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")
        print(f"img_size = {self.img_size}")

        # generate the coordinates of the rectangle
        xx, yy= rectangle_perimeter(start=(a, b), end=(c, d), shape=(self.img_size, self.img_size))

        rectangle_outline = xx, yy
        img = self.get_empty_image()
        img[rectangle_outline] = 1
        print(img.shape)
        img = self.rotate_image_random(img)
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

    def donut(self):
        """
        Returns the image coordinates of an elliptical donut.
        """
        # Define a grid
        xx, yy = np.mgrid[: self.img_size, : self.img_size]
        cx = self.rng.integers(0, self.img_size)
        cy = self.rng.integers(0, self.img_size)

        # Define the width of the donut.
        width = self.rng.uniform(self.img_size / 3, self.img_size)

        # Give the donut some elliptical nature.
        e0 = self.rng.uniform(0.1, 5)
        e1 = self.rng.uniform(0.1, 5)

        # Use the forumula for an ellipse.
        ellipse = e0 * (xx - cx) ** 2 + e1 * (yy - cy) ** 2

        donut = (ellipse < (self.img_size + width)) & (
            ellipse > (self.img_size - width)
        )

        return donut
    
    def stickman(self):
        """
        Returns the image coordinates of a stickman.
        """
        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(0, self.img_size / 3, size=2)
        c, d = self.rng.integers(self.img_size / 2, self.img_size, size=2)

        # Flip a coin to see if slope of line is + or -.
        coin_flip = self.rng.integers(low=0, high=2)
        # Use a skimage.draw method to draw the line.
        if coin_flip:
            xx, yy, _ = line_aa(a, b, c, d)
        else:
            xx, yy, _ = line_aa(a, d, c, b)

        line = xx, yy

        # draw a circle around the end of the line
        circle = circle_perimeter_aa(c, d, self.rng.integers(0, self.img_size / 3))

    
    def arrow(self):
        return None