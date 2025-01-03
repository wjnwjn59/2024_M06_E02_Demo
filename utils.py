import torch
import numpy as np
import os
import gdown

from PIL import Image
from config import AppConfig

def download_model():
    # https://drive.google.com/file/d/1iR7itLAZN4eV6yjgMQoEB7Ovl5tfAiJf/view?usp=sharing
    vqa_id = '1iR7itLAZN4eV6yjgMQoEB7Ovl5tfAiJf'
    unzip_dest = 'weights'
    os.makedirs(unzip_dest, exist_ok=True)

    gdown.download(id=vqa_id, 
                   output=AppConfig().model_weights_path, 
                   quiet=False,
                   fuzzy=True)


def predict(image, question, model, text_tokenizer, img_feature_extractor, idx2label, device="cpu"):
    question_processed = text_tokenizer(
        question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    img_processed = img_feature_extractor(images=image, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_processed, question_processed)
        pred_idx = torch.argmax(output, dim=1).item()
    
    predicted_label = idx2label[pred_idx]

    return predicted_label