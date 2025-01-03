import torch
import streamlit as st
import base64
from PIL import Image
from config import AppConfig
from utils import predict
from components.streamlit_footer import footer
from vqa_model import VQAModel, TextEncoder, VisualEncoder, Classifier 
from transformers import AutoTokenizer, ViTImageProcessor
# Set page configuration
st.set_page_config(
    page_title="AIO2024 Module06 Project - Visual Question Answering",
    page_icon='static/aivn_favicon.png',
    layout="wide"
)


# Load class names
answer_classes = AppConfig().get_answer_classes()

# Load the VQA model
@st.cache_resource
def load_vqa_model():
    # Initialize model components
    text_encoder = TextEncoder().to(AppConfig().device)
    visual_encoder = VisualEncoder().to(AppConfig().device)
    classifier = Classifier(
        hidden_size=256,
        dropout_prob=0.2,
        n_classes=len(answer_classes)
    ).to(AppConfig().device)

    # Load the full model
    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        classifier=classifier
    ).to(AppConfig().device)

    # Load weights
    checkpoint = torch.load(AppConfig().model_weights_path, map_location=AppConfig().device)
    model.visual_encoder.load_state_dict(checkpoint['visual_encoder_state_dict'])
    model.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    return model

# Load tokenizers and processors
@st.cache_resource
def load_processors():
    img_feature_extractor = ViTImageProcessor.from_pretrained(AppConfig().img_encoder_id)
    text_tokenizer = AutoTokenizer.from_pretrained(AppConfig().text_encoder_id)
    return img_feature_extractor, text_tokenizer

img_feature_extractor, text_tokenizer = load_processors()

vqa_model = load_vqa_model()

def main():
    # UI Layout
    col1, col2 = st.columns([0.8, 0.2], gap='large')

    with col1:
        st.title('AIO2024 - Module06 - Visual Question Answering')

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

    # Input: Upload an image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Input: Enter a question
    question = st.text_input("Enter your question about the image:")

    prediction_placeholder = st.empty()  # Placeholder for prediction result

    # Default image if none is uploaded
    default_image_path = "static/default_image.jpg"
    image = Image.open(uploaded_file) if uploaded_file else Image.open(default_image_path)

    if st.button("Answer Question"):
        if question.strip() == "":
            st.error("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                predicted_answer = predict(
                    image=image,
                    question=question,
                    model=vqa_model,
                    text_tokenizer=text_tokenizer,  # Assume this is imported
                    img_feature_extractor=img_feature_extractor,  # Assume this is imported
                    idx2label={idx: label for idx, label in enumerate(answer_classes)},
                    device=AppConfig().device
                )

            # Display prediction
            prediction_placeholder.success(f"Predicted Answer: {predicted_answer}")

    # Display the image
    st.image(image, caption="Uploaded Image" if uploaded_file else "Default Image", use_column_width=True)

    footer()

if __name__ == "__main__":
    main()