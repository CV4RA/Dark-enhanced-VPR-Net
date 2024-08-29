# data/dark_place_loader.py
import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder('path_to_train_dataset', transform=transform)
    val_dataset = datasets.ImageFolder('path_to_val_dataset', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
