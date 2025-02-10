import torch
import torch.nn as nn
import torch.nn.functional as F

'''
To use this initialize the model class and pass the model to the get optimizer along with hyperparameters
        model = KMNISTModel()
        optimizer = get_optimizer(model, optimizer_name=opt, hyperparameters=hyperparamers)
'''

class KMNISTModel(nn.Module):
    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10):
        super(KMNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flattening the 28x28 input image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # Softmax with log for stability


def get_optimizer(model, optimizer_name="adam", lr=0.001, momentum=0.9, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999), alpha=0.99, amsgrad=False, centered=False):
    """
    Returns the optimizer based on the given name and hyperparameters.
    """
    if optimizer_name.lower() == "adamw":
          return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps, betas=betas, amsgrad=amsgrad)
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps, betas=betas, amsgrad=amsgrad)
    elif optimizer_name.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, alpha=alpha, eps=eps, weight_decay=weight_decay, centered=centered)
    else:
        raise ValueError("Unsupported optimizer. Choose from 'adamw', 'adam', or 'rmsprop'.")
