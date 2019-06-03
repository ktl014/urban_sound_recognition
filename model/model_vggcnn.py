"""Initialize CNN model for Sound Recognition"""
# Standard dist imports

# Third party imports
import torch
import torch.nn as nn
import torchvision.models as models

# Project level imports

# Module level constants

class UrbanCNN(nn.Module):
    __names__ = {'VGG', 'AlexNet'}

    def __init__(self, net,num_of_classes=10):
        """ Initialize UrbanCNN

        Ex:
        >>    model = UrbanCNN(net='VGG', num_of_classes=10)
        >>    input = torch.rand(1, 3, 224, 224)
        >>    output = model(input)
        >>    print(output, output.shape)

        Args:
            net (str):  Network architecture
            num_of_classes (int): Number of classes
        """
        super(UrbanCNN, self).__init__()
        assert net in UrbanCNN.__names__

        self.num_of_classes = num_of_classes

        if net == 'VGG':
            self.model = self.get_vgg_arch(self.num_of_classes)
        else:
            self.model = self.get_alexnet_arch(self.num_of_classes)

    def forward(self, x):
        y = self.model(x)
        return y

    def get_alexnet_arch(self, num_of_classes):
        # Initialize pretrained model
        model = models.alexnet(pretrained=True)
        # Modify softmax layer to number of classes
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-2],
                                         nn.Linear(4096, num_of_classes))
        return model

    def get_vgg_arch(self, num_of_classes):
        # Initialize pretrained model
        model = models.vgg16(pretrained=True)
        # Modify softmax layer to number of classes
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-2],
                                         nn.Linear(4096, num_of_classes))
        return model
if __name__ == '__main__':
    model = UrbanCNN(net='VGG', num_of_classes=10)
    input = torch.rand(1, 3, 224, 224)
    output = model(input)
    print(output, output.shape)