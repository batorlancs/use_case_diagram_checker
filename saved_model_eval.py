import torch
from utils import display_boxes, display_masks_rcnn, show, get_maskrcnn
from lightning import ObjectDetection_DM
from torchvision.utils import make_grid

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_maskrcnn(num_classes=3, pretrained=True).to(device)
state_dict = torch.load("models/model_version__2024_02_14___00_31_39_.pth")
model.load_state_dict(state_dict)
model.eval()

dataset = ObjectDetection_DM(
    train_val_size=20,
    img_size=200,
    shapes_per_image=(1, 2),
    class_probs=(1, 1, 0, 0, 0),
    target_masks=True,
    batch_size=4
)

dataset.setup("fit")
dataloader = iter(dataset.train_dataloader())

with torch.no_grad():
    while True:
        try:
            imgs, targets = next(dataloader)
            display_imgs = imgs.type(torch.uint8)  # Convert display_imgs to uint8
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(imgs)
            original = display_masks_rcnn(display_imgs, targets, dataset.class_map)
            original = display_boxes(original, targets, dataset.class_map)
            original_grid = make_grid(original, nrow=5)
            show(original)
            predicted = display_masks_rcnn(display_imgs, preds, dataset.class_map)
            predicted = display_boxes(predicted, preds, dataset.class_map)
            predicted_grid = make_grid(predicted, nrow=5)
            show(predicted)
        except StopIteration:
            break