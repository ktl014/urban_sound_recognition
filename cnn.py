""" """
# Standard dist imports

# Third party imports
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

# Project level imports

# Module level constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        model = models.alexnet(pretrained=True)
        modules = list(model.children())[:-1]
        in_features = 4096
        self.model = nn.Sequential(*modules)
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(9216, in_features),
                                        nn.BatchNorm1d(in_features,
                                                       momentum=0.01))

    def forward(self, input):
        with torch.no_grad():
            features = self.model(input)

        features = features.view(features.size(0), -1)
        features = self.classifier(features)
        return features

if __name__ == '__main__':
    