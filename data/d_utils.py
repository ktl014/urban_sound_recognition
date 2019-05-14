"""Dataset utility functions"""
# Standard dist imports

# Third party imports
from torch.autograd import Variable

# Project level imports

# Module level constants

def to_cuda(item, computing_device, label=False):
    """ Typecast item to cuda()
    Wrapper function for typecasting variables to cuda() to allow for
    flexibility between different types of variables (i.e. long, float)
    Loss function usually expects LongTensor type for labels, which is why
    label is defined as a bool.
    Computing device is usually defined in the Trainer()
    Args:
        item: Desired item. No specific type
        computing_device (str): Desired computing device.
        label (bool): Flag to convert item to long() or float()
    Returns:
        item
    """
    if label:
        item = Variable(item.to(computing_device)).long()
    else:
        item = Variable(item.to(computing_device)).float()
    return item