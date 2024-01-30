import torch
import numpy as np
from skimage.draw import line_aa, circle_perimeter_aa


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

    def __init__(self, img_size: int, rng: np.random.Generator):

        self.img_size = img_size
        self.rng = rng

        return None

    def rectangle(self):
        """
        Returns the image coordinates of a rectangle.
        """
        # Min and max rectangle height.
        a = self.rng.integers(self.img_size * 3 / 20, self.img_size * 7 / 20)
        b = self.rng.integers(self.img_size * 3 / 20, self.img_size * 7 / 20)

        # Initial coordinates of the rectangle.
        xx, yy = np.where(np.ones((a, b)) == 1)

        # Place the rectangle randomly in the image.
        cx = self.rng.integers(0 + a, self.img_size - a)
        cy = self.rng.integers(0 + b, self.img_size - b)

        rectangle = xx + cx, yy + cy

        return rectangle

    def line(self):
        """
        Returns the image coordinates of a line.
        """
        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(low=0, high=self.img_size, size=2)
        c, d = self.rng.integers(low=0, high=self.img_size, size=2)

        # print (a,b,c,d)
        print(f"(a,b) = ({a},{b}), (c,d) = ({c},{d})")

        # Flip a coin to see if slope of line is + or -.
        # coin_flip = self.rng.integers(low=0, high=2)
        # # Use a skimage.draw method to draw the line.
        # if coin_flip:
        xx, yy, _ = line_aa(a, b, c, d)
        # else:
        #     xx, yy, _ = line_aa(a, d, c, b)

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


class CV_DS_Base(torch.utils.data.Dataset):
    """
    Base class for a set of PyTorch computer vision datasets. This class
    contains all of the attributes and methods common to all datasets
    in this package.
    Alone this base class has no functionality. The utility of these datasets
    is that they enable the user to test cv models with very small and
    simple images with tunable complexity. It also requires no downloading
    of images and one can scale the size of the datasets easily.

    Args:
        ds_size (int): number of images in dataset.
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy random number generator.
        class_map (Dict[Dict]): the class map must contain keys (0,1,2,3) 
            and contain names "background", "rectangle", "line", and "donut". 
            "gs_range" specifies the upper and lower bound of the
            grayscale values (0, 255) used to color the shapes.
            "target_color" can be used by visualization tools to assign
            a color to masks and boxes. Note that class 0 is reserved for
            background in most instance seg models, so one can rearrange 
            the class assignments of different shapes but 0 must correspond 
            to "background". The utility of this Dict is to enable the user
            to change target colors, class assignments, and shape 
            intensities. A valid example:    
            class_map={
            0: {"name": "background","gs_range": (200, 255),"target_color": (255, 255, 255),},
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)}}.   
    """

    def __init__(
        self,
        ds_size,
        img_size,
        shapes_per_image,
        class_probs,
        rand_seed,
        class_map,
    ):

        if img_size <= 20:
            raise ValueError(
                "Different shapes are hard to distinguish for images of shape (3, 20, 20) or smaller."
            )

        if sorted(list(class_map.keys())) != sorted([0, 1, 2, 3]):
            raise ValueError("Dict class_map must contain keys 0,1,2,3.")

        self.ds_size = ds_size
        self.img_size = img_size
        self.rand_seed = rand_seed
        self.shapes_per_image = shapes_per_image
        self.class_probs = np.array(class_probs) / np.array(class_probs).sum()
        self.class_map = class_map

        self.rng = np.random.default_rng(self.rand_seed)

        self.draw = Draw(self.img_size, self.rng)

        self.class_ids = np.array([1, 2, 3])
        self.num_shapes_per_img = self.rng.integers(
            low=self.shapes_per_image[0],
            high=self.shapes_per_image[1] + 1,
            size=self.ds_size,
        )

        self.chosen_ids_per_img = [self.rng.choice(
            a=self.class_ids, size=num_shapes, p=self.class_probs
        ) for num_shapes in self.num_shapes_per_img]

        self.imgs, self.targets = [], []

        return None

    def __getitem__(self, idx):

        return self.imgs[idx], self.targets[idx]

    def __len__(self):

        return len(self.imgs)

    def draw_shape(self, class_id):
        """
        Draws the shape with the associated class_id.
        """

        if self.class_map[class_id]["name"] == "rectangle":
            shape = self.draw.rectangle()
        elif self.class_map[class_id]["name"] == "line":
            shape = self.draw.line()
        elif self.class_map[class_id]["name"] == "donut":
            shape = self.draw.donut()
        elif self.class_map[class_id]["name"] == "stickman":
            shape = self.draw.stickman()
        elif self.class_map[class_id]["name"] == "arrow":
            shape = self.draw.arrow()

        else:
            raise ValueError(
                "You must include rectangle, donut, and line in your class_map."
            )

        return shape
