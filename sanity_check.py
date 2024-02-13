import torch
from utils import get_maskrcnn

img_size = 16
x = torch.rand((2, 3, img_size,img_size))

model = get_maskrcnn(num_classes = 2, pretrained = True)
model.eval()
output = model(x)

masks_0_shape = output[0]["masks"].shape
boxes_0_shape = output[0]["boxes"].shape

print(f"Input shape:\n {x.shape} \n" )
print(f"Mask for image 0 shape: \n {masks_0_shape}")
print(f"Boxes for image 0 shape: \n {boxes_0_shape}")
# print(f"Mask RCNN Output:\n {output}")