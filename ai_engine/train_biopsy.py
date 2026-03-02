import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import copy

from utils import get_dataloaders

import logging

# Configuration
dataset_path = '../data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
models_dir = 'saved_models'
logs_dir = '../logs'
batch_size = 32
num_epochs_phase1 = 10   # Training top layers
num_epochs_phase2 = 40   # Fine-tuning base layers
learning_rate = 1e-4

# Setup Logging
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "train_biopsy.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_acc=0.0):
    """
    Standard PyTorch training loop
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            loop = tqdm(dataloader, total=len(dataloader), desc=f"{phase.capitalize()}")
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save checkpoint
                checkpoint_path = os.path.join(models_dir, 'biopsy_resnet50.pth')
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"New best model saved to {checkpoint_path} with Acc: {best_acc:.4f}")

    logger.info(f'Best Validation Accuracy: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def main():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    logger.info("Loading BreaKHis Dataset...")
    # Using the updated generic get_dataloaders from utils.py
    # ResNet-50 standard input size is 224x224
    train_loader, val_loader, num_classes, class_names = get_dataloaders(
        dataset_path, 
        target_size=(224, 224), 
        batch_size=batch_size
    )
    
    if train_loader is None:
        logger.error("Data loading failed. Please ensure the BreaKHis dataset is placed in the correct directory:")
        logger.error(f"Expected: {os.path.abspath(dataset_path)}")
        return

    logger.info(f"Classes found ({num_classes}): {class_names}")
    
    # -------------------------------------------------------------
    # Build ResNet-50 Model (PyTorch)
    # -------------------------------------------------------------
    logger.info("Building ResNet-50 Model...")
    
    # 1. Load pre-trained model on ImageNet
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # 2. Freeze the base model (Phase 1)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Modify the final classification layer
    num_ftrs = model.fc.in_features
    
    # ResNet uses GlobalAveragePooling implicitly before the fc layer
    # We replace it with Dropout + Linear for our classes
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Send model to GPU/CPU
    model = model.to(device)
    
    # Loss function for classification
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------------------------------------
    # Phase 1: Train ONLY the top classification layer
    # -------------------------------------------------------------
    logger.info("\nStarting Phase 1: Training custom top layer (base model frozen)...")
    
    # Optimize only the parameters of the newly added fc layer
    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    model, best_val_acc = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer_phase1, 
        num_epochs_phase1
    )
    
    # -------------------------------------------------------------
    # Phase 2: Fine-tuning ResNet Block 4
    # -------------------------------------------------------------
    logger.info("\nStarting Phase 2: Fine-tuning the last residual block...")
    
    # Instead of unfreezing the whole 50 layers (which might overfit quickly),
    # we only unfreeze layer4 (the last block of ResNet)
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Optimizer with very low learning rate for fine-tuning
    # We pass only the parameters that require gradients
    optimizer_phase2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-5
    )
    
    # Continue training and tracking best accuracy
    model, _ = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer_phase2, 
        num_epochs_phase2,
        best_acc=best_val_acc 
    )
    
    logger.info("\nBreaKHis Training completely finished!")

if __name__ == '__main__':
    main()
