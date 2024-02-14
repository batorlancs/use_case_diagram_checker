from helper_functions.engine import train_one_epoch, evaluate
from utils import get_maskrcnn
from lightning import ObjectDetection_DM
import torch
from datetime import datetime

print("PyTorch Version: ", torch.__version__)
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
class_probs = (1,1,0,0,0)
num_classes = torch.tensor(class_probs).sum().item() + 1
print("num_classes: ", num_classes)

# use our dataset and defined transformations
dataset = ObjectDetection_DM(
    train_val_size=1000,
    img_size = 200,
    rand_seed = 123456,
    shapes_per_image = (1,2), 
    class_probs = class_probs,
    target_masks = True,
    batch_size = 2
)

dataset.setup("fit")
dataset.setup("test")

# define training and validation data loaders
data_loader = dataset.train_dataloader()
data_loader_test = dataset.test_dataloader()

# get the model using our helper function
model = get_maskrcnn(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.0001,
    momentum=0.9,
    weight_decay=0.001
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,
    gamma=0.1
)


num_epochs = 3

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)
    with torch.no_grad():
        for imgs, targets in data_loader_test:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model.eval()
            output = model(imgs)
            print("-"*80)
            print(output)
            print("-"*80)

print("That's it!")
print("Saving model...")
# create models directory if it does not exist
import os
if not os.path.exists("models"):
    os.makedirs("models")

path = f"models/model_version_{datetime.now().strftime('_%Y_%m_%d___%H_%M_%S_')}.pth"
torch.save(model.state_dict(), path)
print("Model saved at: ", path)
