from lightning import ObjectDetection_DM
from utils import show, display_boxes, display_masks_rcnn
from torchvision.utils import make_grid

batch_size = 3

instance_seg_dm = ObjectDetection_DM(
    train_val_size=100,
    train_val_split=[.9,.1],
    test_size=12,
    img_size=200,
    rand_seed=3456,
    shapes_per_image=(1, 4),
    class_probs=(1, 1, 1, 1, 1),
    target_masks=True,
    batch_size=batch_size,
)

instance_seg_dm.setup(stage="fit")
dataiter = iter(instance_seg_dm.train_dataloader())

while True:
    try:
        imgs, targets = next(dataiter)
        res = [imgs[i] for i in range(len(imgs))]
        res = display_boxes(res, targets, instance_seg_dm.class_map, fill=True)
        res = display_masks_rcnn(res, targets, instance_seg_dm.class_map, threshold=.1)
        res = make_grid(res, nrow=len(imgs))
        show(res)

    except StopIteration:
        break