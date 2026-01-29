import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMScheduler
from tqdm import tqdm
import os

from network import UNet


def train_diffusion_model(
    num_epochs=50,
    batch_size=128,
    learning_rate=1e-4,
    num_timesteps=1000,
    device=None,
    save_dir='output'
):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    model = UNet(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=128,
        text_embedding_dim=128,
        num_classes=10
    ).to(device)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_timesteps,
        beta_schedule="squaredcos_cap_v2"
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Number of timesteps: {num_timesteps}")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            timesteps = torch.randint(
                0, num_timesteps, (images.shape[0],), device=device
            ).long()
            
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            optimizer.zero_grad()
            noise_pred = model(noisy_images, timesteps, labels)
            
            loss = criterion(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    final_model_path = os.path.join(save_dir, 'model_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    return model, noise_scheduler


if __name__ == '__main__':
    model, scheduler = train_diffusion_model(
        num_epochs=50,
        batch_size=128,
        learning_rate=1e-4,
        num_timesteps=1000
    )
