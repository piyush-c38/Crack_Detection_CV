import torch
from torch.utils.data import DataLoader, random_split
from dataset import CrackDataset
from model import PSPNet
from train import train

def main():
    image_dir = "data/images"
    mask_dir = "data/masks"
    dataset = CrackDataset(image_dir, mask_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = PSPNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()