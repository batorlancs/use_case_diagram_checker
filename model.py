import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes=4, pretrained=True):

    # Load an instance segmentation model pre-trained on COCO
    if pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT")
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None)

    # Get number of input features for the classifier
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
