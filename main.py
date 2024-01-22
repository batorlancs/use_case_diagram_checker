import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

from model import get_model_instance_segmentation


# img_size = 16
# x = torch.rand((2, 3, img_size,img_size))

# model = get_model_instance_segmentation(num_classes = 2, pretrained = True)
# model.eval()
# output = model(x)

# masks_0_shape = output[0]["masks"].shape

# print(f"Input shape:\n {x.shape} \n" )
# print(f"Mask for image 0 shape: \n {masks_0_shape}")
# print(f"Mask RCNN Output:\n {output}")
