import torch
import numpy as np
from draw_class import Draw
from utils import show

img_size = 500
rng = np.random.default_rng()

draw = Draw(img_size, rng)
# line1 = draw.line()
# line2 = draw.line()
# rec1 = draw.rectangle()
# rec2 = draw.rectangle()
# donut1 = draw.donut()
# donut2 = draw.donut()

test = draw.rectangle_outline()


# create an empty image
img = np.zeros((img_size, img_size))

# draw shapes
# img[line1] = 1
# img[line2] = 1
# img[rec1] = .2
# img[rec2] = .2
# img[donut1] = .4
# img[donut2] = .4

img[test] = 1

# convert to torch tensor
img = torch.from_numpy(img)
print(img.shape)

show(img, figsize=(10, 10))

img.shape, img