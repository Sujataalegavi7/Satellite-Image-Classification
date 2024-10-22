import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import  models
import matplotlib.pyplot as plt

# 1. Preprocessing Function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# 2. Load the pre-trained model
def load_model(model_path):
    model = models.resnet50(pretrained=False)  # Set pretrained to False
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust the output layer to match your classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# 3. Predict function
def predict_image(image_path, model, class_names):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return class_names[preds[0]]

# 4. Visualize the image and prediction
def visualize_prediction(image_path, prediction):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Predicted: {prediction}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Path to your saved model
    model_path = r'C:\Users\Sudha Shukla\Desktop\sic\model\best_model.pth'

    # Define class names (these should match the ones you used in training)
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 
                   'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    # Load the model
    model = load_model(model_path)
    
    # Insert path to the image you want to test
    test_image_path = r'C:\Users\Sudha Shukla\Downloads\AnnualCrop_1.jpg'  # Replace with the path to the test image
    
    # Perform prediction
    prediction = predict_image(test_image_path, model, class_names)
    
    # Visualize the result
    visualize_prediction(test_image_path, prediction)
