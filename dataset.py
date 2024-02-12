import torch
import numpy as np

from draw import Draw


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
        class_probs (Tuple[float, float, float, float, float]): relative probability of
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

        if sorted(list(class_map.keys())) != sorted([0, 1, 2, 3, 4, 5]):
            raise ValueError("Dict class_map must contain keys 0,1,2,3,4,5.")

        self.ds_size = ds_size
        self.img_size = img_size
        self.rand_seed = rand_seed
        self.shapes_per_image = shapes_per_image
        self.class_probs = np.array(class_probs) / np.array(class_probs).sum()
        self.class_map = class_map

        self.rng = np.random.default_rng(self.rand_seed)

        self.draw = Draw(self.img_size, self.rng)

        self.class_ids = np.array([1, 2, 3, 4, 5])
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

    def draw_shape(self, class_id) -> torch.Tensor:
        """
        Draws the shape with the associated class_id.
        """

        if self.class_map[class_id]["name"] == "rectangle":
            shape = self.draw.rectangle()
        elif self.class_map[class_id]["name"] == "line":
            shape = self.draw.line()
        elif self.class_map[class_id]["name"] == "ellipse":
            shape = self.draw.ellipse()
        elif self.class_map[class_id]["name"] == "stickman":
            shape = self.draw.stickman()
        elif self.class_map[class_id]["name"] == "arrow":
            shape = self.draw.dashed_arrow()

        else:
            raise ValueError(
                "You must include rectangle, line, ellipse, stickman, or arrow in the class_map."
            )

        return shape
