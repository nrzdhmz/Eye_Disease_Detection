import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "dataset_split"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5  

# Data augmentation + normalization
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                        std=[0.229, 0.224, 0.225])  
])

classes_to_keep = ['cataract', 'normal']

def filter_dataset(dataset, classes_to_keep):
    idxs = [i for i, (_, label) in enumerate(dataset.samples) if dataset.classes[label] in classes_to_keep]
    dataset.samples = [dataset.samples[i] for i in idxs]
    dataset.targets = [dataset.targets[i] for i in idxs]
    return dataset

# Load datasets
train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
test_data = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

# Filter to keep only selected classes
train_data = filter_dataset(train_data, classes_to_keep)
val_data = filter_dataset(val_data, classes_to_keep)
test_data = filter_dataset(test_data, classes_to_keep)

# Update class names manually
class_names = classes_to_keep
print("Classes:", class_names)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Handle class imbalance with weighted loss
class_counts = Counter(train_data.targets)
total = sum(class_counts.values())
weights_loss = [total / class_counts[i] for i in range(len(class_names))]
weights_loss = torch.tensor(weights_loss).float().to(device)
criterion = nn.CrossEntropyLoss(weight=weights_loss)

# Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block and FC
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Replace final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Training loop with early stopping
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_data)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            val_correct += (preds == labels).sum().item()
    val_acc = val_correct / len(val_data)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model for testing
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(inputs.cpu())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Visualize some misclassified images
wrong_preds = [(img, p, t) for img, p, t in zip(all_images, all_preds, all_labels) if p != t]

def show_image(img_tensor, pred, true):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[pred]}, Actual: {class_names[true]}")
    plt.axis("off")
    plt.show()

for img, pred, true in wrong_preds[:5]:
    show_image(img, pred, true)
