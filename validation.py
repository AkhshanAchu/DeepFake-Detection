import torch
import torch.nn as nn
from tqdm import tqdm
import os
from models import classifier_block
from utils import get_data_loaders
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def validate_model(model_path='best_model.pth', test_dataset_path=None):
    """
    Validate the trained deepfake detection model
    
    Args:
        model_path (str): Path to the saved model
        test_dataset_path (str): Path to test dataset (optional, uses default if None)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    model = classifier_block(n_blocks=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}")
    
    # Set up data loaders
    if test_dataset_path is None:
        # Use default test path from training script
        test_dataset_path = r"C:\Users\shivn\Documents\SP Cup\validation"
    
    print(f"📁 Loading test data from: {test_dataset_path}")
    
    try:
        _, _, test_loader = get_data_loaders(
            train_path="",  # Not needed for validation
            test_path=test_dataset_path,
            batch_size=32,
            max_train_samples=0  # No training data needed
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"📊 Test dataset size: {len(test_loader)} batches")
    
    # Validation metrics
    correct = 0
    total = 0
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    print("🚀 Starting validation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Validating")):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    
    print(f"\n🎯 Validation Results:")
    print(f"   📈 Accuracy: {accuracy:.2f}%")
    print(f"   📉 Average Loss: {avg_loss:.4f}")
    print(f"   ✅ Correct Predictions: {correct}/{total}")
    
    # Detailed classification metrics
    print(f"\n📋 Detailed Classification Report:")
    class_names = ['Real', 'Fake']  # Assuming binary classification
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('🎭 Deepfake Detection - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('validation_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Confusion matrix saved as 'validation_confusion_matrix.png'")
    
    # Per-class accuracy
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    if cm.size == 4:  # Binary classification
        real_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
        fake_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n🎭 Per-Class Performance:")
        print(f"   🎪 Real Images Accuracy: {real_accuracy*100:.2f}%")
        print(f"   🤖 Fake Images Accuracy: {fake_accuracy*100:.2f}%")
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   🎯 Precision: {precision*100:.2f}%")
        print(f"   🔍 Recall: {recall*100:.2f}%")
        print(f"   ⚖️ F1-Score: {f1_score*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'confusion_matrix': cm
    }


def validate_on_custom_dataset(model_path='best_model.pth', custom_test_path=None):
    """
    Validate model on a custom dataset (like DeepWild)
    
    Args:
        model_path (str): Path to the saved model
        custom_test_path (str): Path to custom test dataset
    """
    print("🌟 Custom Dataset Validation")
    print("="*50)
    
    if custom_test_path:
        results = validate_model(model_path, custom_test_path)
        print(f"\n🎉 Custom dataset validation completed!")
        return results
    else:
        print("❌ Please provide custom_test_path for custom dataset validation")
        return None


if __name__ == "__main__":
    print("🎭 Deepfake Detection Model Validation")
    print("="*50)
    
    # Standard validation
    print("\n1️⃣ Standard Validation:")
    results = validate_model()
    
    # Custom dataset validation example (uncomment and modify path as needed)
    # print("\n2️⃣ Custom Dataset Validation (DeepWild):")
    # custom_results = validate_on_custom_dataset(
    #     model_path='best_model.pth',
    #     custom_test_path=r"path\to\deepwild\dataset"
    # )
    
    print("\n🎊 Validation completed successfully!")