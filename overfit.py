import torch
from lightning import ObjectDetection_DM
from utils import get_maskrcnn, apply_score_cut, display_boxes, display_masks_rcnn, show
from torchvision.utils import make_grid

def overfit(imgs, targets, model, optimizer,  device,  epochs=100): 
    
    model = model.to(device)
    model.train()

    # Formatting for input to model. 
    imgs_normed = imgs.float()
    imgs_normed = imgs_normed.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    for epoch in range(epochs):
        loss_dict = model(imgs_normed, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if epoch %25 == 0:
            print(f"epoch: {epoch}")
            print(f"loss {losses:.4f}\n")

    return None


instance_seg_dm = ObjectDetection_DM(img_size = 100,
                                       rand_seed = 3456,
                                       shapes_per_image = (1,1), 
                                       class_probs = (1,1,0,0,0),
                                       target_masks = True,
                                       batch_size = 3)

instance_seg_dm.setup(stage = "fit")
dataiter = iter(instance_seg_dm.train_dataloader())

imgs, targets = next(dataiter)

# print(f"Input shape:\n {imgs.shape} \n" )

# print(f"Mask RCNN target for img 0:\n ")
# for key, val in targets[0].items():
#     print(f"\n{key}:\n {val}")
#     print(f"\n{key}.shape :\n {val.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_maskrcnn(num_classes = 3, pretrained = True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
overfit(imgs, targets, model, optimizer,  device,  epochs=100)

# evalutation
model.eval()

# Formatting for input to model. 
imgs_normed = imgs.float()
imgs_normed = imgs_normed.to(device)
preds = model(imgs_normed)
preds = apply_score_cut(preds, score_threshold=0.25)

threshold = .5
result_image = [imgs[i] for i in range(len(preds))]

target_image = display_boxes(result_image, targets, instance_seg_dm.class_map, fill = True)
target_image = display_masks_rcnn(target_image, targets, instance_seg_dm.class_map)

pred_image = display_boxes(result_image, preds, instance_seg_dm.class_map)
pred_image = display_masks_rcnn(pred_image, preds, instance_seg_dm.class_map, threshold = threshold)

target_grid = make_grid(target_image)
pred_grid = make_grid(pred_image)
show(target_grid)
show(pred_grid)