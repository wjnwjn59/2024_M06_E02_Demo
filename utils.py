import torch
import numpy as np
import os
import gdown

from config import AppConfig

def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0

    return normalized_img


def download_model():
    # https://drive.google.com/file/d/1U_mH6rVWO6mxKBWG0iczjgSpaLeFdVyz/view?usp=drive_link
    resnet_id = '1U_mH6rVWO6mxKBWG0iczjgSpaLeFdVyz'

    # https://drive.google.com/file/d/1IL3IEK-BXo8HYGTe665QuMOnw3dN51tr/view?usp=sharing
    densenet_id = '1IL3IEK-BXo8HYGTe665QuMOnw3dN51tr'
    unzip_dest = 'weights'
    os.makedirs(unzip_dest, exist_ok=True)

    gdown.download(id=resnet_id, 
                   output=AppConfig().resnet_weights_path, 
                   quiet=False,
                   fuzzy=True)

    gdown.download(id=densenet_id, 
                   output=AppConfig().densenet_weights_path, 
                   quiet=False,
                   fuzzy=True)


def predict(image, model, class_names, device="cpu"):
    """
    Perform inference on the input image and return the predicted class.
    """
    image_tensor = transform(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]
    return predicted_class
