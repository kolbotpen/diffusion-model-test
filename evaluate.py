import torch
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import os
from network import UNet
from tqdm import tqdm


def generate_digits_with_visualization(
    model_path='output/model_final.pt',
    digit_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    num_samples_per_digit=5,
    num_inference_steps=1000,
    device=None,
    save_dir='output/generated_samples',
    show_interval=50
):
    os.makedirs(save_dir, exist_ok=True)
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    model = UNet(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=128,
        text_embedding_dim=128,
        num_classes=10
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_inference_steps,
        beta_schedule="squaredcos_cap_v2"
    )
    
    print(f"\n{'='*60}")
    print(f"Generating {num_samples_per_digit} samples per digit with live visualization")
    print(f"Total denoising steps: {num_inference_steps}")
    print(f"Display updates every {show_interval} steps")
    print(f"{'='*60}\n")
    
    plt.ion()
    
    with torch.no_grad():
        for digit in digit_labels:
            print(f"\n🔢 Generating {num_samples_per_digit} samples for digit: {digit}")
            
            labels = torch.tensor([digit] * num_samples_per_digit, device=device)
            images = torch.randn((num_samples_per_digit, 1, 28, 28), device=device)
            
            noise_scheduler.set_timesteps(num_inference_steps)
            timesteps_list = noise_scheduler.timesteps.tolist()
            
            fig, axes = plt.subplots(1, num_samples_per_digit, figsize=(num_samples_per_digit * 3, 3))
            if num_samples_per_digit == 1:
                axes = [axes]
            fig.suptitle(f'Generating Digit {digit} - LIVE ({num_samples_per_digit} samples)', 
                        fontsize=16, fontweight='bold')
            
            for step_idx, t in enumerate(tqdm(timesteps_list, desc=f"  Denoising")):
                timesteps = torch.full((num_samples_per_digit,), t, device=device, dtype=torch.long)
                noise_pred = model(images, timesteps, labels)
                images = noise_scheduler.step(noise_pred, t, images).prev_sample
                
                if step_idx % show_interval == 0 or step_idx == len(timesteps_list) - 1:
                    imgs_normalized = (images + 1) / 2
                    imgs_normalized = torch.clamp(imgs_normalized, 0, 1)
                    
                    for idx in range(num_samples_per_digit):
                        current_img = imgs_normalized[idx].cpu().squeeze().numpy()
                        axes[idx].clear()
                        axes[idx].imshow(current_img, cmap='gray')
                        axes[idx].axis('off')
                        progress_percent = (step_idx / len(timesteps_list)) * 100
                        axes[idx].set_title(f'Sample {idx+1}\nStep {step_idx}\n{progress_percent:.0f}%', fontsize=10)
                    
                    plt.tight_layout()
                    plt.pause(0.01)
            
            final_images = (images + 1) / 2
            final_images = torch.clamp(final_images, 0, 1)
            
            for idx in range(num_samples_per_digit):
                axes[idx].clear()
                axes[idx].imshow(final_images[idx].cpu().squeeze().numpy(), cmap='gray')
                axes[idx].axis('off')
                axes[idx].set_title(f'Sample {idx+1}\n Complete', fontsize=11, fontweight='bold')
            
            fig.suptitle(f' Digit {digit} Complete - {num_samples_per_digit} Samples Generated', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.pause(2.0)
            
            grid_path = os.path.join(save_dir, f'digit_{digit}_samples.png')
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f" Saved: {grid_path}")
    
    plt.ioff()
    
    print(f"\n{'='*60}")
    print("Creating combined grid of all digits...")
    all_images = []
    
    with torch.no_grad():
        for digit in tqdm(digit_labels, desc="Generating final grid"):
            labels = torch.tensor([digit], device=device)
            image = torch.randn((1, 1, 28, 28), device=device)
            
            noise_scheduler.set_timesteps(num_inference_steps)
            for t in noise_scheduler.timesteps:
                timesteps = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model(image, timesteps, labels)
                image = noise_scheduler.step(noise_pred, t, image).prev_sample
            
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)
            all_images.append(image.cpu().squeeze().numpy())
    
    # Plot grid
    fig, axes = plt.subplots(1, len(digit_labels), figsize=(len(digit_labels) * 2, 2))
    fig.suptitle('All Generated Digits (0-9)', fontsize=16, fontweight='bold')
    
    for idx, (digit, img) in enumerate(zip(digit_labels, all_images)):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'{digit}', fontsize=14)
    
    plt.tight_layout()
    grid_path = os.path.join(save_dir, 'all_digits_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved combined grid: {grid_path}")
    
    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"Check '{save_dir}/' for saved images")
    print(f"{'='*60}\n")


def generate_digits(
    model_path='output/model_final.pt',
    digit_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    num_samples_per_digit=5,
    num_inference_steps=1000,
    device=None,
    save_dir='output/generated_samples'
):
    os.makedirs(save_dir, exist_ok=True)
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    model = UNet(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=128,
        text_embedding_dim=128,
        num_classes=10
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_inference_steps,
        beta_schedule="squaredcos_cap_v2"
    )
    
    print(f"Generating {num_samples_per_digit} samples for each digit: {digit_labels}")
    print(f"Using {num_inference_steps} inference steps")
    
    with torch.no_grad():
        for digit in digit_labels:
            print(f"\nGenerating digit: {digit}")
            
            labels = torch.tensor([digit] * num_samples_per_digit, device=device)
            
            image_shape = (num_samples_per_digit, 1, 28, 28)
            images = torch.randn(image_shape, device=device)
            
            noise_scheduler.set_timesteps(num_inference_steps)
            
            for t in noise_scheduler.timesteps:
                timesteps = torch.full((num_samples_per_digit,), t, device=device, dtype=torch.long)
                noise_pred = model(images, timesteps, labels)
                
                images = noise_scheduler.step(noise_pred, t, images).prev_sample
            
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            
            fig, axes = plt.subplots(1, num_samples_per_digit, figsize=(num_samples_per_digit * 2, 2))
            if num_samples_per_digit == 1:
                axes = [axes]
            
            for idx in range(num_samples_per_digit):
                img = images[idx].cpu().squeeze().numpy()
                axes[idx].imshow(img, cmap='gray')
                axes[idx].axis('off')
                axes[idx].set_title(f'Digit {digit}')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'digit_{digit}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
    
    print("\nGenerating combined grid of all digits")
    all_images = []
    
    with torch.no_grad():
        for digit in digit_labels:
            labels = torch.tensor([digit], device=device)
            image = torch.randn((1, 1, 28, 28), device=device)
            
            noise_scheduler.set_timesteps(num_inference_steps)
            for t in noise_scheduler.timesteps:
                timesteps = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model(image, timesteps, labels)
                image = noise_scheduler.step(noise_pred, t, image).prev_sample
            
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)
            all_images.append(image.cpu().squeeze().numpy())
    
    fig, axes = plt.subplots(1, len(digit_labels), figsize=(len(digit_labels) * 2, 2))
    for idx, (digit, img) in enumerate(zip(digit_labels, all_images)):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'{digit}', fontsize=16)
    
    plt.tight_layout()
    grid_path = os.path.join(save_dir, 'all_digits_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid: {grid_path}")
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    print("\n Starting generation with LIVE on-screen display!\n")
    print(" Watch 5 samples transform from noise to digit in real-time!\n")
    
    generate_digits_with_visualization(
        model_path='output/model_final.pt',
        digit_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        num_samples_per_digit=5,
        num_inference_steps=1000,
        show_interval=50
    )


