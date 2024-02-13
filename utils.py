import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def show(imgs, figsize=(10.0, 10.0), grayscale=False):
    """Displays a single image or list of images. Taken more or less from
    the pytorch docs:
    https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#visualizing-a-grid-of-images

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        figsize (Tuple[float, float]): size of figure to display.

    Returns:
        None
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img, mode="L" if grayscale else None)
        axs[0, i].imshow(np.asarray(img), cmap="gray" if grayscale else None)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None


def display_boxes(imgs, target_pred_dict, class_map, width=1, fill=False):
    """
    Takes a list of images and a list of target or prediction dictionaries
    of the same len and overlays bounding boxes onto the images.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        target_pred_dict (List[Dict[torch.Tensor]]): predictions or targets
            formatted according to the torchvision implimentation of
            FasterRCNN and MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            A valid example ("name" not necessary):
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        fill (bool): if True the inside of the bounding boxes will be
            filled with color.

    Returns:
        result_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
            overlaid bounding boxes.
    """
    num_imgs = len(imgs)
    result_imgs = [
        draw_bounding_boxes(
            imgs[i],
            target_pred_dict[i]["boxes"].int(),
            fill=fill,
            colors=[
                class_map[j.item()]["target_color"]
                for j in target_pred_dict[i]["labels"]
            ],
            width=width,
        )
        for i in range(num_imgs)
    ]

    return result_imgs


def apply_score_cut(preds, score_threshold=0.5):
    """
    Takes a list of prediction dictionaries (one for each image) and cuts
    out all instances whose score is below the score threshold.

    Args:
        preds (List[Dict[torch.Tensor]]): predictions as output by the
            torchvision implimentation of MaskRCNN or FasterRCNN. The 
            scores are in the range (0,1) and signify the certainty of 
            the model for that instance.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html
        score_threshold (float): the threshold to apply to the identified
            objects. If an instance is below the score_threshold it will
            be removed from the score_thresholded_preds dictionary.

    Returns:
        score_thresholded_preds (List[Dict[torch.Tensor]]): predictions
            that exceed score_threshold.
    """
    score_thresholded_preds = [
        {key: value[pred["scores"] > score_threshold]
            for key, value in pred.items()}
        for pred in preds
    ]

    return score_thresholded_preds


def threshold_pred_masks(preds, threshold=0.5):
    """
    Takes a list of prediction dictionaries (one for each image) and
    thresholds the soft masks, returning a list of prediction dictionaries
    with thresholded (boolean) masks.

    Args:
        preds (List[Dict[torch.Tensor]]): predictions as output by the
            torchvision implimentation of MaskRCNN. The masks consist of
            probabilities (torch.float32) in the range (0,1) for each pixel.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

    Returns:
        thresholded_preds (List[Dict[torch.Tensor]]): predictions with
            boolean (torch.bool) masks.
    """

    thresholded_preds = [
        {**pred, "masks": (pred["masks"] > threshold).squeeze(dim=1)} for pred in preds
    ]

    return thresholded_preds


def display_masks_rcnn(imgs, target_pred_dict, class_map, threshold=0.5, alpha=0.4):
    """
    Takes a list of images and a list of target or prediction dictionaries
    of the same len and overlays segmentation masks onto the images.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        target_pred_dict (List[Dict[torch.Tensor]]): predictions or targets
            formatted according to the torchvision implimentation of
            FasterRCNN and MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            A valid example ("name" not necessary):
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        threshold (float): threshold applied to soft masks. In range (0-1).

        alpha (float): transparnecy of masks. In range (0-1).

    Returns:
        result_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
            overlaid segmentation masks.
    """
    num_imgs = len(imgs)

    if target_pred_dict[0]["masks"].dtype == torch.float32:
        target_pred_dict = threshold_pred_masks(target_pred_dict, threshold)

    result_imgs = [
        draw_segmentation_masks(
            imgs[i],
            target_pred_dict[i]["masks"],
            alpha=alpha,
            colors=[
                class_map[j.item()]["target_color"]
                for j in target_pred_dict[i]["labels"]
            ],
        )
        for i in range(num_imgs)
    ]

    return result_imgs

def get_maskrcnn(num_classes=4, pretrained=True):
    """A function for loading the PyTorch implimentation of MaskRCNN.
    To not have predictor changed at all set num_classes = -1.
    See here for documentation on the input and output specifics:
    https://pytorch.org/vision/0.12/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html

    Args:
        num_classes (int): number of output classes desired.
        pretrained (bool): whether or not to load a model pretrained on the COCO dataset. 
    """

    if pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    if num_classes != -1:

        # Get number of input features for the classifier.
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained box predictor head with a new one.
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Now get the number of input features for the mask classifier.
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        hidden_layer = 256
        # Replace the pre-trained mask predictor head with a new one.
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    return model
