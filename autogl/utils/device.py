import torch

def get_device(device):
    assert isinstance(device, (str, torch.device)), "Only support device of str or torch.device, get {} instead".format(device)
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)
