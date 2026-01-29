#!/usr/bin/env python3
"""
Evaluate handwriting diffusion model - Generate handwriting from text prompts

Usage:
    python3 evaluate_handwriting.py handwriting
    python3 evaluate_handwriting.py "hello world"
    python3 evaluate_handwriting.py test --steps 100 --model output/handwriting/model_final.pt
"""

import torch
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import os
import sys
import argparse
from network_handwriting import UNetHandwriting
from tqdm import tqdm


def text_to_indices(text, max_len=32):
    """Convert text string to character indices"""
    indices = [ord(c) if ord(c) < 128 else 32 for c in text[:max_len]]
    # Pad with spaces (ASCII 32)
    while len(indices) < max_len:
        indices.append(32)
    return torch.tensor([indices], dtype=torch.long)


def generate_handwriting(
    text,
    model_path='output/handwriting/model_final.pt',
    num_inference_steps=1000,
    device=None,
    save_path=None,
    show_process=False
):
    """Generate handwriting image from text prompt"""
    
    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Generating handwriting for: '{text}'")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = UNetHandwriting(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=256,
        text_embedding_dim=256,
        vocab_size=128,
        max_seq_len=32
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_inference_steps,
        beta_schedule="squaredcos_cap_v2"
    )
    
    # Convert text to indices
    text_indices = text_to_indices(text).to(device)
    
    # Start from random noise
    image = torch.randn(1, 1, 64, 256, device=device)
    
    # Setup visualization if requested
    if show_process:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 3))
    
    # Denoising loop
    noise_scheduler.set_timesteps(num_inference_steps)
    
    print(f"Running {num_inference_steps} denoising steps...")
    with torch.no_grad():
        for step_idx, t in enumerate(tqdm(noise_scheduler.timesteps)):
            timesteps = torch.full((1,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(image, timesteps, text_indices)
            
            # Denoise
            image = noise_scheduler.step(noise_pred, t, image).prev_sample
            
            # Show intermediate results
            if show_process and step_idx % 50 == 0:
                img_normalized = (image[0, 0].cpu() + 1) / 2
                img_normalized = torch.clamp(img_normalized, 0, 1)
                
                ax.clear()
                ax.imshow(img_normalized.numpy(), cmap='gray')
                ax.set_title(f"'{text}' - Step {step_idx}/{num_inference_steps}")
                ax.axis('off')
                plt.pause(0.01)
    
    # Denormalize final image
    image = (image + 1) / 2
    image = torch.clamp(image, 0, 1)
    final_image = image[0, 0].cpu().numpy()
    
    # Display and save result
    if show_process:
        plt.ioff()
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(final_image, cmap='gray')
    ax.set_title(f"Generated Handwriting: '{text}'", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save image
    if save_path is None:
        # Create default save path
        os.makedirs('output/handwriting/generated', exist_ok=True)
        safe_filename = "".join(c if c.isalnum() else "_" for c in text)
        save_path = f'output/handwriting/generated/{safe_filename}.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Image saved to: {save_path}")
    
    # Show the image
    plt.show()
    
    return final_image, save_path


def generate_multiple_samples(
    text,
    model_path='output/handwriting/model_final.pt',
    num_samples=4,
    num_inference_steps=1000,
    device=None,
    save_path=None
):
    """Generate multiple variations of the same text"""
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Generating {num_samples} variations of: '{text}'")
    
    # Load model
    model = UNetHandwriting(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=256,
        text_embedding_dim=256,
        vocab_size=128,
        max_seq_len=32
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_inference_steps,
        beta_schedule="squaredcos_cap_v2"
    )
    
    text_indices = text_to_indices(text).to(device)
    
    # Generate multiple samples
    images = []
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")
            
            # Start from different random noise each time
            image = torch.randn(1, 1, 64, 256, device=device)
            
            noise_scheduler.set_timesteps(num_inference_steps)
            
            for t in tqdm(noise_scheduler.timesteps, desc=f"Sample {i+1}"):
                timesteps = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model(image, timesteps, text_indices)
                image = noise_scheduler.step(noise_pred, t, image).prev_sample
            
            # Denormalize
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)
            images.append(image[0, 0].cpu().numpy())
    
    # Display all samples
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, num_samples * 2))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"Variation {idx+1}: '{text}'", fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        os.makedirs('output/handwriting/generated', exist_ok=True)
        safe_filename = "".join(c if c.isalnum() else "_" for c in text)
        save_path = f'output/handwriting/generated/{safe_filename}_variations.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Image saved to: {save_path}")
    
    plt.show()
    
    return images, save_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate handwriting from text using trained diffusion model'
    )
    parser.add_argument(
        'text',
        type=str,
        nargs='?',
        default='handwriting',
        help='Text to generate as handwriting (default: "handwriting")'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='output/handwriting/model_final.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of denoising steps (default: 1000, lower is faster but lower quality)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for generated image'
    )
    parser.add_argument(
        '--show-process',
        action='store_true',
        help='Show denoising process in real-time'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        help='Number of variations to generate (default: 1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use for generation'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("\nPlease train the model first using:")
        print("  python3 train_handwriting.py")
        print("\nOr download a pre-trained model and specify its path with --model")
        sys.exit(1)
    
    print("="*60)
    print("Handwriting Generation")
    print("="*60)
    
    # Generate samples
    if args.samples > 1:
        generate_multiple_samples(
            text=args.text,
            model_path=args.model,
            num_samples=args.samples,
            num_inference_steps=args.steps,
            device=args.device,
            save_path=args.output
        )
    else:
        generate_handwriting(
            text=args.text,
            model_path=args.model,
            num_inference_steps=args.steps,
            device=args.device,
            save_path=args.output,
            show_process=args.show_process
        )
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)


if __name__ == '__main__':
    # If run with command line args, use them
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior: generate "handwriting"
        print("No arguments provided. Generating default text: 'handwriting'")
        print("Usage: python3 evaluate_handwriting.py <text>")
        print()
        generate_handwriting('handwriting')
