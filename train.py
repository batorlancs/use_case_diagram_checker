from lightning import ObjectDetection_DM
from lightning2 import InstanceSeg_LM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
from datetime import datetime

instance_seg_dm = ObjectDetection_DM(train_val_size=500,
                                     train_val_split=[.9, .1],
                                     test_size=12,
                                     img_size=100,
                                     batch_size=4,
                                     shapes_per_image=(1, 1),
                                     class_probs=(1, 1, 0, 0, 0),
                                     target_masks=True)

# Create Instance of Lightning Module. num_classes = num_shapes + 1 (for background)
instance_seg_lm = InstanceSeg_LM(num_classes=3, lr=1e-4, pretrained=True)

# Create callback for ModelCheckpoints.
checkpoint_callback = ModelCheckpoint(filename='{epoch:02d}', save_top_k=30, monitor="Loss/train_loss", every_n_epochs=1)

# Define Logger.
logger = TensorBoardLogger(
    "tb_logs", name="instance_segmentation", log_graph=False)

# Set device.
device = "gpu" if torch.cuda.is_available() else "cpu"

# Create an instance of a Trainer.
# trainer = pl.Trainer(fast_dev_run=True,  accelerator = device)
# trainer = pl.Trainer(overfit_batches=1,  accelerator = device)
trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], accelerator=device, max_epochs=5, log_every_n_steps=1, check_val_every_n_epoch=1)

# Fit.
trainer.fit(instance_seg_lm, instance_seg_dm)

# Save model
folder_path = "models"
# add date and time to file name
path = f"{folder_path}/Model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
torch.save(instance_seg_lm.state_dict(), path)
print("model saved...")