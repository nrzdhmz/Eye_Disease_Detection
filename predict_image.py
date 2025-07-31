import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224
class_names = ['cataract', 'normal']  
model_path = "best_model.pth"         
image_path = "images_to_predict/my_eye1.png"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# === Load Model ===
def load_model(model_path):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# === Predict Function ===
def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = torch.argmax(outputs, 1).item()
    return class_names[pred_idx], img

# === Display Prediction ===
def show_prediction(img, prediction):
    plt.imshow(img)
    plt.title(f"Predicted: {prediction}")
    plt.axis("off")
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    elif not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    else:
        model = load_model(model_path)
        prediction, img = predict_image(model, image_path)
        print(f"Prediction: {prediction}")
        show_prediction(img, prediction)
