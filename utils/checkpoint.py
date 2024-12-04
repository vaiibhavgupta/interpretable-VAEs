import torch
import os

def save_checkpoint(model, optimizer, epoch, target_dataset):
    path = f'checkpoints/vae_checkpoint_{target_dataset}.pth'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, target_dataset):
    path = f'checkpoints/vae_checkpoint_{target_dataset}.pth'

    if os.path.exists(path):
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')    
        return checkpoint['epoch']
    return 0