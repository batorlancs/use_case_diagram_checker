import torch
import numpy as np
from dataset import CV_DS_Base
from torchvision.ops import masks_to_boxes


class ObjectDetection_DS(CV_DS_Base):
    """
    Self contained PyTorch Dataset for testing object detection and
    instance segmentation models.
    Note that the specifics of the target formatting is adherent to the
    requirements of the torchvision MaskRCNN and FasterRCNN implimentations.
    That said, this dataset should work with any object detection or
    instance segmentation model that requires the same target formatting
    (such as YOLO).
    See the MaskRCNN documentation (linked below) for more details on the
    formatting of the targets.
    https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

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
        target_masks (bool): whether or not the target dictionaries should
            contain boolean masks for each object instance. Masks are not
            necessary to train FasterRCNN or other object detection models
            but are necessary to train instance segmentation models such
            as MaskRCNN.
    """

    def __init__(
        self,
        ds_size=100,
        img_size=256,
        shapes_per_image=(1, 3),
        class_probs=(1, 1, 1, 1, 1),
        rand_seed=12345,
        class_map={
            0: {
                "name": "background",
                "gs_range": (200, 255),
                "target_color": (255, 255, 255),
            },
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "ellipse", "gs_range": (0, 100), "target_color": (0, 0, 255)},
            4: {"name": "stickman", "gs_range": (0, 100), "target_color": (0, 255, 255)},
            5: {"name": "arrow", "gs_range": (0, 100), "target_color": (255, 255, 0)},
        },
        target_masks=False,
    ):

        super().__init__(
            ds_size, img_size, shapes_per_image, class_probs, rand_seed, class_map
        )

        self.target_masks = target_masks
        self.imgs, self.targets = self.build_imgs_and_targets()

    def build_imgs_and_targets(self):
        """
        Builds images and targets for object detection and instance
        segmentation.

        Returns:
            imgs (torch.UInt8Tensor[ds_size, 3, img_size, img_size]): images
                containing different shapes. The images are gray-scale
                (each layer of the first (color) dimension is identical).
                This makes it easier to visualize targets and predictions.
            targets (List[Dict[torch.Tensor]]): list of dictionaries of
                length ds_size. Each image has an associated target
                dictionary that contains the following (note that N = number
                of instances/shapes in a given img):
                - boxes (torch.FloatTensor[N, 4]): the ground-truth boxes
                in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                0 <= y1 < y2 <= H.
                - labels (torch.Int64Tensor[N]): the class label for each
                ground-truth box.
                - masks (torch.UInt8Tensor[N, H, W]): the segmentation
                binary masks for each instance. Not included if
                target_masks = False.

        """
        imgs = []
        targets = []

        for idx in range(self.ds_size):

            chosen_ids = self.chosen_ids_per_img[idx]
            num_shapes = self.num_shapes_per_img[idx]

            # Creating an empty noisy img.
            img = self.rng.integers(
                self.class_map[0]["gs_range"][0],
                self.class_map[0]["gs_range"][1],
                (self.img_size, self.img_size),
            )
            # Empty target dictionary to be filled.
            target = {}
            masks = np.zeros((num_shapes, self.img_size, self.img_size))

            # Filling the noisy img with shapes and building masks.
            for i, class_id in enumerate(chosen_ids):
                shape = self.draw_shape(class_id)
                gs_range = self.class_map[class_id]["gs_range"]

                threshold = .1
                indexes = torch.where(shape.squeeze() > threshold)

                # set img to shape img where shape img has higher value then img
                img[indexes] = self.rng.integers(gs_range[0], gs_range[1])
                masks[i][indexes] = 1

            # Convert from np to torch and assign appropriate dtypes.
            img = torch.from_numpy(img)
            img = img.unsqueeze(dim=0).repeat(3, 1, 1).type(torch.ByteTensor)

            masks = torch.from_numpy(masks).bool()
            chosen_ids = torch.from_numpy(chosen_ids).long()

            # Fill in the target dictionary.
            if self.target_masks:
                target["masks"] = masks
            boxes = masks_to_boxes(masks)

            # check if boxes height or width is 0
            # if so, set the box to the entire image
            for i, box in enumerate(boxes):
                if box[2] - box[0] == 0:
                    boxes[i][0] = min(boxes[i][0]-1, 0)
                    boxes[i][2] = max(boxes[i][2]+1, self.img_size)
                    # print("ADJUSTED X")
                if box[3] - box[1] == 0:
                    boxes[i][1] = min(boxes[i][1]-1, 0)
                    boxes[i][3] = max(boxes[i][3]+1, self.img_size)
                    # print("ADJUSTED Y")

            target["boxes"] = boxes
            target["area"] = self.boxes_area(target["boxes"])
            target["labels"] = chosen_ids
            target["image_id"] = torch.tensor([idx])
            target["iscrowd"] = torch.zeros((num_shapes,), dtype=torch.int64)

            targets.append(target)
            imgs.append(img)

        # Turn a list of imgs with shape (3, H, W) of len ds_size to a tensor
        # of shape (ds_size, 3, H, W)
        imgs = torch.stack(imgs)

        return imgs, targets

    def boxes_area(self, boxes):
        """
        Returns the area of a bounding box with [x1, y1, x2, y2] format, with
        0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        """
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return area
