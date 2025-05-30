import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# -----------------------------------------------
# 1. Define number of classes for each crop
classes_per_crop = {
    'pepper': 2,
    'potato': 3,
    'tomato': 10
}

# -----------------------------------------------
# 2. Model file paths (adjust these paths as per your setup)
model_paths = {
    'pepper': 'models/pepper_model.pth',
    'potato': 'models/potato_model.pth',
    'tomato': 'models/tomato_model.pth'
}

# -----------------------------------------------
# 3. Raw class names as per your model training labels
class_names = {
    'pepper': ['Class1', 'Class2'],
    'potato': ['Early_blight', 'Late_blight', 'Healthy'],
    'tomato': ['Tomato_Class1', 'Tomato_Class2', 'Tomato_Class3', 'Tomato_Class4',
               'Tomato_Class5', 'Tomato_Class6', 'Tomato_Class7', 'Tomato_Class8',
               'Tomato_Class9', 'Tomato_Class10']
}

# -----------------------------------------------
# 4. User-friendly disease names mapping
disease_name_map = {
    'pepper': {
        'Class1': "Pepper - Bacterial Spot",
        'Class2': "Pepper - Healthy"
    },
    'potato': {
        'Early_blight': "Potato - Early Blight",
        'Late_blight': "Potato - Late Blight",
        'Healthy': "Potato - Healthy"
    },
    'tomato': {
        'Tomato_Class1': "Tomato - Early Blight",
        'Tomato_Class2': "Tomato - Late Blight",
        'Tomato_Class3': "Tomato - Leaf Mold",
        'Tomato_Class4': "Tomato - Septoria Leaf Spot",
        'Tomato_Class5': "Tomato - Spider Mites",
        'Tomato_Class6': "Tomato - Target Spot",
        'Tomato_Class7': "Tomato - Yellow Leaf Curl Virus",
        'Tomato_Class8': "Tomato - Tomato Mosaic Virus",
        'Tomato_Class9': "Tomato - Healthy",
        'Tomato_Class10': "Tomato - Another Disease"  # Replace with actual name if known
    }
}

# -----------------------------------------------
# 5. Function to create the model architecture with correct output classes
def get_model(num_classes):
    model = models.resnet18(pretrained=False)  # Use ResNet18 without pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust final layer for num_classes
    return model

# -----------------------------------------------
# 6. Load model for selected crop with caching for efficiency
@st.cache_resource
def load_model(crop):
    num_classes = classes_per_crop[crop]
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_paths[crop], map_location=torch.device('cpu')))
    model.eval()
    return model

# -----------------------------------------------
# 7. Image preprocessing pipeline - resizing, tensor conversion, normalization
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 as expected by ResNet
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# -----------------------------------------------
# 8. Prediction function - returns raw class label
def predict(image, model, crop):
    input_tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)  # Get index of highest score
    return class_names[crop][predicted.item()]  # Map index to class name

# -----------------------------------------------
# 9. Streamlit UI starts here
st.title("Plant Disease Detection")

# Dropdown to select the crop
crop = st.selectbox("Select Crop", ['pepper', 'potato', 'tomato'])

# File uploader to upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and crop:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model for selected crop
    model = load_model(crop)

    # Get the raw prediction
    prediction = predict(image, model, crop)

    # Map raw class label to user-friendly disease name
    friendly_name = disease_name_map[crop].get(prediction, "Unknown Disease")

    # Display the prediction nicely
    st.success(f'Prediction: {friendly_name}')
