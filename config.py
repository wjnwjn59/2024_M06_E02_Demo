import torch

class AppConfig:
    def __init__(self):
        self.model_weights_path = "weights/vqa_model.pth"
        self.img_encoder_id = "google/vit-base-patch16-224"
        self.text_encoder_id = "roberta-base"
        self.answer_classes_file = "classes/answer_space.txt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_answer_classes(self):
        """
        Reads the answer classes from a file.
        Returns:
            List of answer class labels.
        """
        with open(self.answer_classes_file, "r") as f:
            return [line.strip() for line in f.readlines()]
