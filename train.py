import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import BoundaryComboLoss
from utils import iou_score, dice_score
from model import PSPNet

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BoundaryComboLoss()
    
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            total_iou, total_dice = 0, 0
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                total_iou += iou_score(outputs, masks).item()
                total_dice += dice_score(outputs, masks).item()
            print(f"Val IoU: {total_iou / len(val_loader):.4f}, Val Dice: {total_dice / len(val_loader):.4f}")