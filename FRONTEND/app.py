import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, render_template, request, jsonify
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Define device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model
model = models.resnet18(pretrained=False)  # No pre-trained weights, loading custom model
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification (Fake/Real)
model.load_state_dict(torch.load("deepfake_model_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Save image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image).squeeze().item()
    
    # Apply sigmoid to get probability
    probability = torch.sigmoid(torch.tensor(output)).item()
    
    # Determine classification
    result = "Fake" if probability > 0.5 else "Real"
    confidence = round(probability * 100 if result == "Fake" else (1 - probability) * 100, 2)

    return jsonify({"result": result, "confidence": confidence, "image_path": image_path})

if __name__ == "__main__":
    app.run(debug=True)
