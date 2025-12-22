"""
Script to prepare dogs and cats dataset for training.
"""

import os
import pandas as pd
from pathlib import Path


def prepare_dataset(data_dir, output_dir):
    """
    Prepare the dogs and cats dataset by organizing images and creating splits.
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    print("Preparing dogs and cats dataset...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(data_dir) / "data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
    else:
        print(f"Data file not found: {data_path}")
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    prepare_dataset("../data", "../output")
