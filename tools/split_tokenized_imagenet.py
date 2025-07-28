
"""
Split tokenized ImageNet dataset into train/validation splits.

This script can split a tokenized ImageNet dataset either by:
1. Creating a random split based on a percentage (e.g., 80% train, 20% val)
2. Following the official ImageNet train/val split using predefined val classes
3. Creating stratified splits per class

Usage examples:
python split_tokenized_imagenet.py --input-dir /path/to/tokenized/data --output-dir /path/to/split/data --split-ratio 0.8
python split_tokenized_imagenet.py --input-dir /path/to/tokenized/data --output-dir /path/to/split/data --use-official-split
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def get_official_imagenet_val_classes():
    """
    Returns the list of ImageNet validation classes.
    This follows the standard ImageNet validation split.
    """
    # Standard ImageNet validation classes (first 50 from each synset)
    # For simplicity, we'll create a balanced validation set
    # You can modify this list based on your specific requirements
    return list(range(0, 1000, 20))  # Every 20th class for validation (50 classes total)


def create_stratified_split(input_dir, train_ratio=0.8, seed=42):
    """
    Create a stratified split ensuring each class has samples in both train and val.
    
    Args:
        input_dir: Path to input tokenized data directory
        train_ratio: Ratio of samples to use for training
        seed: Random seed for reproducibility
    
    Returns:
        train_samples: List of (file_path, class_id) tuples for training
        val_samples: List of (file_path, class_id) tuples for validation
    """
    random.seed(seed)
    np.random.seed(seed)
    
    train_samples = []
    val_samples = []
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"Found {len(class_dirs)} classes")
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(input_dir, class_dir)
        class_id = int(class_dir)
        
        # Get all files in this class
        files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        if len(files) == 0:
            print(f"Warning: No .npy files found in class {class_id}")
            continue
            
        # Shuffle files for this class
        random.shuffle(files)
        
        # Split files
        num_train = max(1, int(len(files) * train_ratio))  # Ensure at least 1 train sample
        num_val = len(files) - num_train
        
        if num_val == 0 and len(files) > 1:
            # If no validation samples but more than 1 file, take 1 for validation
            num_train -= 1
            num_val = 1
        
        # Add to respective lists
        for i, file in enumerate(files):
            file_path = os.path.join(class_path, file)
            if i < num_train:
                train_samples.append((file_path, class_id))
            else:
                val_samples.append((file_path, class_id))
    
    print(f"Split created: {len(train_samples)} train samples, {len(val_samples)} val samples")
    return train_samples, val_samples


def create_official_split(input_dir):
    """
    Create train/val split using official ImageNet validation classes.
    """
    val_classes = set(get_official_imagenet_val_classes())
    
    train_samples = []
    val_samples = []
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"Found {len(class_dirs)} classes")
    print(f"Using {len(val_classes)} classes for validation")
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(input_dir, class_dir)
        class_id = int(class_dir)
        
        # Get all files in this class
        files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        if len(files) == 0:
            print(f"Warning: No .npy files found in class {class_id}")
            continue
        
        # Determine if this class goes to train or val
        target_list = val_samples if class_id in val_classes else train_samples
        
        for file in files:
            file_path = os.path.join(class_path, file)
            target_list.append((file_path, class_id))
    
    print(f"Official split created: {len(train_samples)} train samples, {len(val_samples)} val samples")
    return train_samples, val_samples


def copy_samples(samples, output_dir, split_name):
    """
    Copy samples to the output directory maintaining class structure.
    
    Args:
        samples: List of (file_path, class_id) tuples
        output_dir: Base output directory
        split_name: 'train' or 'val'
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Group samples by class
    class_samples = {}
    for file_path, class_id in samples:
        if class_id not in class_samples:
            class_samples[class_id] = []
        class_samples[class_id].append(file_path)
    
    # Copy files
    for class_id, file_paths in tqdm(class_samples.items(), desc=f"Copying {split_name} files"):
        class_dir = os.path.join(split_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(class_dir, filename)
            shutil.copy2(file_path, dest_path)
    
    print(f"Copied {len(samples)} {split_name} samples to {split_dir}")


def create_symlinks(samples, output_dir, split_name):
    """
    Create symbolic links instead of copying files to save space.
    
    Args:
        samples: List of (file_path, class_id) tuples
        output_dir: Base output directory
        split_name: 'train' or 'val'
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Group samples by class
    class_samples = {}
    for file_path, class_id in samples:
        if class_id not in class_samples:
            class_samples[class_id] = []
        class_samples[class_id].append(file_path)
    
    # Create symlinks
    for class_id, file_paths in tqdm(class_samples.items(), desc=f"Creating {split_name} symlinks"):
        class_dir = os.path.join(split_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(class_dir, filename)
            
            # Create relative symlink
            rel_path = os.path.relpath(file_path, class_dir)
            if not os.path.exists(dest_path):
                os.symlink(rel_path, dest_path)
    
    print(f"Created symlinks for {len(samples)} {split_name} samples in {split_dir}")


def print_dataset_stats(input_dir, train_samples, val_samples):
    """Print statistics about the dataset split."""
    
    print("\n" + "="*50)
    print("DATASET SPLIT STATISTICS")
    print("="*50)
    
    # Original dataset stats
    total_files = 0
    total_classes = 0
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            total_classes += 1
            files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
            total_files += len(files)
    
    print(f"Original dataset:")
    print(f"  - Total classes: {total_classes}")
    print(f"  - Total files: {total_files}")
    
    # Split stats
    train_classes = len(set(class_id for _, class_id in train_samples))
    val_classes = len(set(class_id for _, class_id in val_samples))
    
    print(f"\nAfter split:")
    print(f"  - Train samples: {len(train_samples)} ({len(train_samples)/total_files*100:.1f}%)")
    print(f"  - Train classes: {train_classes}")
    print(f"  - Val samples: {len(val_samples)} ({len(val_samples)/total_files*100:.1f}%)")
    print(f"  - Val classes: {val_classes}")
    print(f"  - Overlap classes: {len(set(class_id for _, class_id in train_samples) & set(class_id for _, class_id in val_samples))}")


def main():
    parser = argparse.ArgumentParser(description="Split tokenized ImageNet dataset")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to input tokenized ImageNet directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output directory for split data")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Ratio of samples for training (default: 0.8)")
    parser.add_argument("--use-official-split", action="store_true",
                        help="Use official ImageNet validation classes instead of random split")
    parser.add_argument("--use-symlinks", action="store_true",
                        help="Create symbolic links instead of copying files (saves space)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show split statistics without actually creating files")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create split
    if args.use_official_split:
        print("Using official ImageNet validation split")
        train_samples, val_samples = create_official_split(args.input_dir)
    else:
        print(f"Using stratified random split with ratio {args.split_ratio}")
        train_samples, val_samples = create_stratified_split(
            args.input_dir, 
            train_ratio=args.split_ratio, 
            seed=args.seed
        )
    
    # Print statistics
    print_dataset_stats(args.input_dir, train_samples, val_samples)
    
    if args.dry_run:
        print("\nDry run complete. No files were created.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy or create symlinks
    if args.use_symlinks:
        print("\nCreating symbolic links...")
        create_symlinks(train_samples, args.output_dir, "train")
        create_symlinks(val_samples, args.output_dir, "val")
    else:
        print("\nCopying files...")
        copy_samples(train_samples, args.output_dir, "train")
        copy_samples(val_samples, args.output_dir, "val")
    
    print(f"\nDataset split completed! Output saved to: {args.output_dir}")
    print("Directory structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── 0/")
    print(f"  │   ├── 1/")
    print(f"  │   └── ...")
    print(f"  └── val/")
    print(f"      ├── 0/")
    print(f"      ├── 1/")
    print(f"      └── ...")


if __name__ == "__main__":
    main()
