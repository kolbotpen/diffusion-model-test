import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from network_handwriting import UNetHandwriting
from dataset_handwriting import create_handwriting_dataloaders


def train_handwriting_diffusion(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    num_timesteps=1000,
    device=None,
    save_dir='output/handwriting',
    data_dir='data/archive/iam_words',
    img_height=64,
    img_width=256
):
    """Train diffusion model for handwriting generation"""
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    
    # Create dataloaders
    print("Loading IAM Handwriting dataset...")
    train_loader, val_loader = create_handwriting_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Create model
    model = UNetHandwriting(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=256,
        text_embedding_dim=256,
        vocab_size=128,
        max_seq_len=32
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_timesteps,
        beta_schedule="squaredcos_cap_v2"
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss function
    criterion = nn.MSELoss()
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Image size: {img_height}x{img_width}")
    print(f"  Save directory: {save_dir}")
    print(f"{'='*60}\n")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, text_indices, _) in enumerate(progress_bar):
            images = images.to(device)
            text_indices = text_indices.to(device)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, num_timesteps, (images.shape[0],), device=device
            ).long()
            
            # Add noise to images
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Forward pass
            optimizer.zero_grad()
            noise_pred = model(noisy_images, timesteps, text_indices)
            
            # Compute loss
            loss = criterion(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, text_indices, _ in val_loader:
                images = images.to(device)
                text_indices = text_indices.to(device)
                
                timesteps = torch.randint(
                    0, num_timesteps, (images.shape[0],), device=device
                ).long()
                
                noise = torch.randn_like(images)
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, text_indices)
                
                loss = criterion(noise_pred, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Generate sample images
            generate_samples(
                model, noise_scheduler, device,
                save_path=os.path.join(save_dir, 'samples', f'epoch_{epoch+1}.png'),
                num_inference_steps=50  # Faster sampling for training
            )
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'model_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    print(f"Training curve saved: {os.path.join(save_dir, 'training_curve.png')}")
    
    return model, noise_scheduler


def generate_samples(model, noise_scheduler, device, save_path, num_inference_steps=50):
    """Generate sample handwriting during training"""
    model.eval()
    
    sample_texts = ['hello', 'world', 'test', 'AI']
    
    with torch.no_grad():
        all_images = []
        
        for text in sample_texts:
            # Convert text to indices
            indices = [ord(c) if ord(c) < 128 else 32 for c in text[:32]]
            while len(indices) < 32:
                indices.append(32)
            text_indices = torch.tensor([indices], device=device, dtype=torch.long)
            
            # Start from random noise
            image = torch.randn(1, 1, 64, 256, device=device)
            
            # Set timesteps
            noise_scheduler.set_timesteps(num_inference_steps)
            
            # Denoise
            for t in noise_scheduler.timesteps:
                timesteps = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model(image, timesteps, text_indices)
                image = noise_scheduler.step(noise_pred, t, image).prev_sample
            
            # Denormalize
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)
            all_images.append(image[0, 0].cpu().numpy())
        
        # Save images
        fig, axes = plt.subplots(1, len(sample_texts), figsize=(15, 3))
        for idx, (img, text) in enumerate(zip(all_images, sample_texts)):
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(text)
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()


if __name__ == '__main__':
    model, scheduler = train_handwriting_diffusion(
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        num_timesteps=1000,
        data_dir='data/archive/iam_words'
    )
