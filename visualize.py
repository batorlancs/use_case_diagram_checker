from utils import display_boxes, display_masks_rcnn, show
from lightning import ObjectDetection_DM
from torchvision.utils import make_grid

img_size = 400
shapes_per_img_lo = 3
shapes_per_img_hi = 5
class_prob_1 = 1
class_prob_2 = 1
class_prob_3 = 1
gs_range_0_lo = 200
gs_range_0_hi = 255
gs_range_1_lo = 0
gs_range_1_hi = 100
rand_seed = 12345
display_size = 30
target_box = True
target_mask = True

instance_seg_dm = ObjectDetection_DM(
    train_val_size=10,
    img_size=img_size,
    train_val_split=(.6, .4),
    shapes_per_image=(shapes_per_img_lo, shapes_per_img_hi),
    class_probs=(class_prob_1, class_prob_2, class_prob_3),
    rand_seed=rand_seed,
    class_map={
        0: {"name": "background", "gs_range": (gs_range_0_lo, gs_range_0_hi), "target_color": (255, 255, 255)},
        1: {"name": "rectangle", "gs_range": (gs_range_1_lo, gs_range_1_hi), "target_color": (255, 0, 0)},
        2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
        3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)},
    },
    target_masks=True,
    dataloader_shuffle={"train": False, "val": False, "test": False},
)

# Visualize and understand some random training images
instance_seg_dm.setup(stage="fit")
dataiter = iter(instance_seg_dm.train_dataloader())

# SimpleMaskRCNN_DS()
imgs, targets = next(dataiter)

result_image = [img for img in imgs]

if target_box:
    result_image = display_boxes(
        result_image, targets, instance_seg_dm.class_map, fill=True)
if target_mask:
    result_image = display_masks_rcnn(
        result_image, targets, instance_seg_dm.class_map)

grid = make_grid(result_image)
show(grid, figsize=(display_size, display_size))
