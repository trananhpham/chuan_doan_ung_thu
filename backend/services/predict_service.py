import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import logging

logger = logging.getLogger(__name__)

# Model File Paths (Absolute path based on this file's location)
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_SERVICE_DIR, "..", "..", "ai_engine", "saved_models")
ULTRASOUND_MODEL_PATH = os.path.abspath(os.path.join(MODELS_DIR, "ultrasound_efficientnet_b3.pth"))
BIOPSY_MODEL_PATH = os.path.abspath(os.path.join(MODELS_DIR, "biopsy_resnet50.pth"))

# Class Names
ULTRASOUND_CLASSES = ['benign', 'malignant', 'normal']
BIOPSY_CLASSES = ['benign', 'malignant']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation (Same as validation transforms in training)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variables to hold models in memory
ultrasound_model = None
biopsy_model = None

def load_models():
    """Loads both deep learning models into memory."""
    global ultrasound_model, biopsy_model
    
    logger.info("Loading PyTorch Models into memory...")
    try:
        # Load Ultrasound Model (EfficientNet-B3)
        if os.path.exists(ULTRASOUND_MODEL_PATH):
            logger.info("Loading Ultrasound Model...")
            ultrasound_model = models.efficientnet_b3(weights=None)
            num_ftrs_us = ultrasound_model.classifier[1].in_features
            ultrasound_model.classifier[1] = nn.Linear(num_ftrs_us, len(ULTRASOUND_CLASSES))
            ultrasound_model.load_state_dict(torch.load(ULTRASOUND_MODEL_PATH, map_location=device, weights_only=True))
            ultrasound_model = ultrasound_model.to(device)
            ultrasound_model.eval()
        else:
            logger.error(f"Ultrasound model not found at {ULTRASOUND_MODEL_PATH}")

        # Load Biopsy Model (ResNet-50)
        if os.path.exists(BIOPSY_MODEL_PATH):
            logger.info("Loading Biopsy Model...")
            biopsy_model = models.resnet50(weights=None)
            num_ftrs_bx = biopsy_model.fc.in_features
            biopsy_model.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs_bx, len(BIOPSY_CLASSES))
            )
            biopsy_model.load_state_dict(torch.load(BIOPSY_MODEL_PATH, map_location=device, weights_only=True))
            biopsy_model = biopsy_model.to(device)
            biopsy_model.eval()
        else:
            logger.error(f"Biopsy model not found at {BIOPSY_MODEL_PATH}")

        logger.info("Model loading complete.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def preprocess_image(image_bytes):
    """Converts raw bytes to a tensor ready for prediction."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = val_transforms(image).unsqueeze(0) # Add batch dimension
        return tensor.to(device)
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return None

def predict_ultrasound(image_bytes):
    """Runs inference for Ultrasound imagery."""
    if ultrasound_model is None:
        return {"error": "Ultrasound model not loaded on server."}
        
    tensor = preprocess_image(image_bytes)
    if tensor is None:
        return {"error": "Invalid image data."}
        
    with torch.no_grad():
        outputs = ultrasound_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class_name = ULTRASOUND_CLASSES[predicted_idx.item()]
    
    # Return exactly formatted response for UI consumption
    return {
        "success": True,
        "prediction": class_name.capitalize(),
        "confidence": round(confidence.item() * 100, 2),
        "details": {ULTRASOUND_CLASSES[i].capitalize(): round(probabilities[i].item() * 100, 2) for i in range(len(ULTRASOUND_CLASSES))}
    }

def predict_biopsy(image_bytes):
    """Runs inference for Biopsy (Histopathology) imagery."""
    if biopsy_model is None:
        return {"error": "Biopsy model not loaded on server."}
        
    tensor = preprocess_image(image_bytes)
    if tensor is None:
        return {"error": "Invalid image data."}
        
    with torch.no_grad():
        outputs = biopsy_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class_name = BIOPSY_CLASSES[predicted_idx.item()]
    
    return {
        "success": True,
        "prediction": class_name.capitalize(),
        "confidence": round(confidence.item() * 100, 2),
        "details": {BIOPSY_CLASSES[i].capitalize(): round(probabilities[i].item() * 100, 2) for i in range(len(BIOPSY_CLASSES))}
    }
