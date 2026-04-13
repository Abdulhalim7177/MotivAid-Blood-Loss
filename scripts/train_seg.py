import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import BloodLossDataset
import os

# -- Config --
EPOCHS = 80
BATCH_SIZE = 16
LR = 1e-4
PATIENCE = 10  # STOP after 10 epochs if no improvement
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- Data (Now with a SEPARATE Validation Set) --
train_ds = BloodLossDataset('dataset/synthetic_train', 'dataset/masks', 'dataset/synthetic_labels.json', mode='train')
val_ds   = BloodLossDataset('dataset/synthetic_val',   'dataset/masks', 'dataset/synthetic_labels.json', mode='val')

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# -- Model --
model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid').to(DEVICE)
loss_fn = smp.losses.DiceLoss('binary') + smp.losses.SoftBCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -- Training with Early Stopping --
best_iou = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    # (Simplified train loop...)
    # [Training logic here...]

    # Validation check
    model.eval()
    val_iou = 0.75 # [Calculated Val IoU...]
    
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'models/seg_best.pt')
        patience_counter = 0  # reset
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        print(f"Stopping Early at Epoch {epoch}. Model has stopped improving.")
        break
