from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import torch.nn as nn
import torch


class CBIR():
    def __init__(self):
        self.model = nn.Sequential(*list(BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k').children())[0:-1]).to('cuda')
        self.preprocess = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    def get_feature(self, batch_of_image):
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch_of_image)
            features = output['pooler_output'] / torch.linalg.norm(output['pooler_output'])
            return features