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
        # st.title(':sparkles: :blue[Stereo Matching] Image Depth Estimation Demo')
        
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

    model_choice = st.selectbox("Select a Model", ["ResNet", "DenseNet"])
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify Image"):
            with st.spinner("Processing..."):
                if model_choice == "ResNet":
                    predicted_class = predict(image, resnet_model, weather_classes, config.device)
                else:
                    predicted_class = predict(image, densenet_model, scenes_classes, config.device)
            st.success(f"Predicted Class: {predicted_class}")

    footer()

if __name__ == "__main__":
    if not os.path.exists(AppConfig.resnet_weights_path) or not os.path.exists(AppConfig.densenet_weights_path):
        download_model()
    main()