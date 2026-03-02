import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset_dir, target_size=(224, 224), batch_size=16, val_split=0.2):
    """
    Creates PyTorch DataLoaders for training and validation.
    Assumes data is organized in folders by class:
    dataset_dir/
      class_1/
      class_2/
      ...
    """
    
    # 1. Define transformations for training (with strong augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness/contrast slightly
        transforms.ToTensor(),
        # Standard ImageNet normalization for pre-trained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # 2. Define transformations for validation (no augmentation, only resize and normalize)
    val_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        print(f"WARNING: Directory {dataset_dir} not found!")
        return None, None, 0, None
    import logging
    logger = logging.getLogger(__name__)
    
    # Custom handling for BreaKHis nested structure or flat BUSI structure
    # BreaKHis is structured: breast/{benign|malignant}/{subclass}/{patient_id}/{magnification}/image.png
    # BUSI is structured: Dataset_BUSI_with_GT/{benign|malignant|normal}/image.png
    samples = []
    classes = []
    
    # Hardcode the expected structure to avoid OS traversal bugs returning non-dirs
    if "BreaKHis" in dataset_dir:
        classes = ['benign', 'malignant']
    elif "BUSI" in dataset_dir:
        classes = ['benign', 'malignant', 'normal']
    else:
        # Fallback for generic datasets
        classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    logger.info(f"Using classes: {classes}")
    
    for cls_name in classes:
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.exists(cls_dir):
            logger.warning(f"Class directory not found: {cls_dir}")
            continue
            
        count = 0
        # Walk through all subdirectories inside the class folder
        for root, _, files in os.walk(cls_dir):
            for file in files:
                if file.endswith('.png') and not ('_mask' in file):
                    path = os.path.join(root, file)
                    samples.append((path, class_to_idx[cls_name]))
                    count += 1
        logger.info(f" - Found {count} valid images for class '{cls_name}'")
                    
    if not samples:
        logger.error("No valid images found. Check the dataset path.")
        print("ERROR: No valid images found. Check the dataset path.")
        return None, None, 0, None
        
    num_classes = len(classes)
    class_names = classes
    
    # Create a lightweight generic Dataset replacing ImageFolder
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            return self.samples[idx]
            
    full_dataset = CustomDataset(samples)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Split dataset randomly
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # For reproducibility
    )
    
    # TRICK: We need different transforms for train and val, but random_split
    # keeps the same underlying full_dataset transforms for both.
    # We override them using a custom wrapper class.
    class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            # Our CustomDataset returns the raw image path directly as the 'x'
            raw_img_path, target = self.subset.dataset.samples[self.subset.indices[index]]
            from PIL import Image
            img = Image.open(raw_img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target
        
        def __len__(self):
            return len(self.subset)

    train_dataset = TransformWrapper(train_dataset, transform=train_transforms)
    val_dataset = TransformWrapper(val_dataset, transform=val_transforms)

    # Create DataLoaders
    # Note: num_workers=0 is safer on Windows. Increase to 2 or 4 on Linux/Colab for speed.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, num_classes, class_names
