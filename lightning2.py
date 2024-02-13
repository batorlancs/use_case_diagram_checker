import torch
import pytorch_lightning as pl
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torch.nn.functional as F

class InstanceSeg_LM(pl.LightningModule):

    def __init__(self, num_classes = 4, pretrained = True, lr = 3e-4,):
        super().__init__()
        # LM Attributes.
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.lr = lr
        # Log hyperparameters. 
        self.save_hyperparameters()
        # Metrics. 
        self.map_bbox = MeanAveragePrecision(iou_type = "bbox", class_metrics = False)
        # self.map_mask = MeanAveragePrecision(iou_type = "segm", class_metrics = False)
        # Mask RCNN model. 
        self.model = self.get_maskrcnn(self.num_classes, self.pretrained)

    def forward(self, imgs):
        self.model.eval()
        imgs_normed = self.norm_imgs(imgs)
        return self.model(imgs_normed)

    def training_step(self, train_batch, batch_idx):
        imgs, targets = train_batch
        imgs_normed = self.norm_imgs(imgs)
        loss_dict = self.model(imgs_normed, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('Loss/train_loss', losses)
        return losses

    def validation_step(self, val_batch, batch_idx):
        imgs, targets = val_batch
        preds = self.forward(imgs)
        self.map_bbox.reset()
        # self.map_mask.reset()
        self.map_bbox.update(preds, targets)
        # self.threshold_pred_masks(preds, threshold = .5 )
        # self.map_mask.update(self.thresholded_preds, targets)
        self.log('mAP_bbox/val', self.map_bbox.compute()["map"])
        # self.log('mAP_segm/val', self.map_mask.compute()["map"])

        return None

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_maskrcnn(self, num_classes, pretrained):
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

    def norm_imgs(self, imgs):
        return imgs.float()

    def threshold_pred_masks(self, preds, threshold = .5): 
        self.thresholded_preds = [{**pred, 'masks': (pred["masks"] > threshold).squeeze(dim = 1)} for pred in preds]
        return None
 