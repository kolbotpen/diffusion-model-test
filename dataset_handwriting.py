import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms


class IAMHandwritingDataset(Dataset):
    """
    Dataset for IAM Handwriting Database
    Expects data structure:
        data/archive/iam_words/
            words/
                a01/
                    a01-000u/
                        a01-000u-00-00.png
                        ...
            words.txt (metadata file)
    """
    def __init__(self, root_dir, split='train', img_height=64, img_width=256, max_text_len=32):
        self.root_dir = root_dir
        self.words_dir = os.path.join(root_dir, 'words')
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_len = max_text_len
        
        # Load metadata
        self.samples = self._load_metadata(split)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
    def _load_metadata(self, split):
        """Load word image paths and corresponding text labels"""
        words_file = os.path.join(self.root_dir, 'words.txt')
        samples = []
        
        if not os.path.exists(words_file):
            print(f"Warning: {words_file} not found. Creating dummy dataset...")
            return self._create_dummy_samples()
        
        with open(words_file, 'r') as f:
            for line in f:
                # Skip comments
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                word_id = parts[0]
                segmentation_result = parts[1]
                
                # Skip failed segmentations
                if segmentation_result != 'ok':
                    continue
                
                text = parts[-1]  # The actual word text (last column)
                
                # Build image path: a01-000u-00 -> a01/a01-000u/a01-000u-00.png
                parts_id = word_id.split('-')
                img_path = os.path.join(
                    self.words_dir,
                    parts_id[0],
                    f"{parts_id[0]}-{parts_id[1]}",
                    f"{word_id}.png"
                )
                
                if os.path.exists(img_path):
                    samples.append({
                        'image_path': img_path,
                        'text': text
                    })
        
        # Simple train/val split (90/10)
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        split_idx = int(0.9 * len(samples))
        
        if split == 'train':
            samples = [samples[i] for i in indices[:split_idx]]
        else:  # validation
            samples = [samples[i] for i in indices[split_idx:]]
        
        print(f"Loaded {len(samples)} {split} samples from IAM dataset")
        return samples
    
    def _create_dummy_samples(self):
        """Create dummy samples for testing when IAM dataset is not available"""
        words = ['hello', 'world', 'test', 'handwriting', 'sample', 
                 'the', 'quick', 'brown', 'fox', 'jumps']
        samples = []
        for _ in range(100):
            word = np.random.choice(words)
            samples.append({
                'image_path': None,
                'text': word
            })
        return samples
    
    def text_to_indices(self, text):
        """Convert text to character indices"""
        # Pad or truncate to max_text_len
        indices = [ord(c) if ord(c) < 128 else 32 for c in text[:self.max_text_len]]
        
        # Pad with spaces (ASCII 32) if needed
        while len(indices) < self.max_text_len:
            indices.append(32)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Load or create image
        if sample['image_path'] and os.path.exists(sample['image_path']):
            try:
                image = Image.open(sample['image_path']).convert('L')  # Grayscale
                image = self.transform(image)
            except Exception as e:
                # If image loading fails, create a blank image
                print(f"Error loading {sample['image_path']}: {e}")
                image = torch.randn(1, self.img_height, self.img_width)
        else:
            # Create dummy image (for testing)
            image = torch.randn(1, self.img_height, self.img_width)
        
        # Convert text to indices
        text_indices = self.text_to_indices(text)
        
        return image, text_indices, text


def create_handwriting_dataloaders(root_dir='data/archive/iam_words', batch_size=32, num_workers=4):
    """Create train and validation dataloaders"""
    train_dataset = IAMHandwritingDataset(root_dir, split='train')
    val_dataset = IAMHandwritingDataset(root_dir, split='val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test the dataset
    dataset = IAMHandwritingDataset('data/archive/iam_words', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        image, text_indices, text = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Text indices shape: {text_indices.shape}")
        print(f"Text: {text}")
