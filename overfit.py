import torch
from lightning import ObjectDetection_DM
from utils import get_maskrcnn, apply_score_cut

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
                                       shapes_per_image = (1,2), 
                                       class_probs = (1,1,0,0,0),
                                       target_masks = True,
                                       batch_size = 3)

instance_seg_dm.setup(stage = "fit")
dataiter = iter(instance_seg_dm.train_dataloader())

imgs, targets = next(dataiter)

print(f"Input shape:\n {imgs.shape} \n" )

print(f"Mask RCNN target for img 0:\n ")
for key, val in targets[0].items():
    print(f"\n{key}:\n {val}")
    print(f"\n{key}.shape :\n {val.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_maskrcnn(num_classes = 3, pretrained = True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
overfit(imgs, targets, model, optimizer,  device,  epochs= 500)

# evalutation
model.eval()

# Formatting for input to model. 
imgs_normed = imgs.float()
imgs_normed = imgs_normed.to(device)
preds = model(imgs_normed)
preds = apply_score_cut(preds, score_threshold=0.25)