import os
import streamlit as st
import base64
from PIL import Image
from cnn_models import load_resnet, load_densenet
from utils import predict, download_model
from config import AppConfig

from components.streamlit_footer import footer

# Set page configuration
st.set_page_config(
    page_title="AIO2024 Module02 Project Image Depth Estimation - AI VIETNAM",
    page_icon='static/aivn_favicon.png',
    layout="wide"
)

# Initialize config
config = AppConfig()

if not os.path.exists(config.resnet_weights_path) or not os.path.exists(config.densenet_weights_path):
    download_model()

# Load class names
weather_classes = config.get_weather_classes()
scenes_classes = config.get_scenes_classes()

# Load models
@st.cache_resource
def get_models():
    densenet = load_densenet(config.densenet_weights_path, 
                             num_classes=len(scenes_classes), 
                             device=config.device)
    resnet = load_resnet(config.resnet_weights_path, 
                         num_classes=len(weather_classes), 
                         device=config.device)

    return densenet, resnet

densenet_model, resnet_model = get_models()

def main():
    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title('AIO2024 - Module06 - Advanced CNN Architectures')
        
    with col2:
        logo_img = open("static/aivn_logo.png", "rb").read()
        logo_base64 = base64.b64encode(logo_img).decode()
        st.markdown(
            f"""
            <a href="https://aivietnam.edu.vn/">
                <img src="data:image/png;base64,{logo_base64}" width="full">
            </a>
            """,
            unsafe_allow_html=True,
        )
        
    st.markdown("Choose a model to classify your image!")

    # Updated dropdown with additional notes
    model_choice = st.selectbox(
        "Select a Model",
        ["ResNet (Weather Image Classification)", "DenseNet (Natural Scene Classification)"]
    )

    # Check model choice and assign the default image and class names accordingly
    if "ResNet" in model_choice:
        default_image_path = "static/rime.jpg"
        model = resnet_model
        classes = weather_classes
        class_file = config.weather_classes_file
    else:
        default_image_path = "static/glacier.jpg"
        model = densenet_model
        classes = scenes_classes
        class_file = config.scenes_classes_file

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    prediction_placeholder = st.empty()  # Placeholder for prediction result

    # Display class names beside the "Classify Image" button
    with open(class_file, "r") as f:
        class_list = f.read().splitlines()
    st.markdown(f"**Class Names for {model_choice.split(' ')[0]}:**")
    st.markdown(", ".join(class_list))

    # Check if a user has uploaded an image or use the default image
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(default_image_path)  # Default image based on model choice

    if st.button("Classify Image"):
        with st.spinner("Processing..."):
            predicted_class = predict(image, model, classes, config.device)

        # Display prediction above the image
        prediction_placeholder.success(f"Predicted Class: {predicted_class}")

    # Display the image (uploaded or default)
    st.image(image, caption="Uploaded Image" if uploaded_file else "Default Image", use_column_width=True)

    footer()

if __name__ == "__main__":
    main()
