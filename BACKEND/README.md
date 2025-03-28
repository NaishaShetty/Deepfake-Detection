# Overview
This backend is built using Flask and serves as the API for the Deepfake Detection system. It loads a pre-trained deep learning model (ResNet18) to classify images as real or deepfake. The API handles image uploads, processes them, and returns predictions.

# Features
* Accepts image uploads via an API endpoint
* Preprocesses images before passing them to the model
* Loads a pre-trained ResNet-18 model for deepfake detection
* Returns a confidence score for real vs. fake classification

# Technologies used
* Flask - Lightweight web framework for handling API requests
* PyTorch - Deep learning framework for model inference
* Torchvision - Provides pre-trained models (ResNet-18) and image transformations
* Werkzeug - Utility for handling file uploads
* Pillow (PIL) - Image processing library
