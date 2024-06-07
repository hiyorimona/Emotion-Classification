import torch
from transformers import AdamW


MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-05
EPSILON = 1e-8
WEIGHT_DECAY = 0.001

def loss_fn(outputs, targets):
    """
    Calculate the binary cross-entropy loss between model predictions and target labels.

    Parameters:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Target labels.

    Returns:
        torch.Tensor: Binary cross-entropy loss.
    """
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
def optimizer(model):
    """
    Creates an AdamW optimizer for training the provided model.

    Parameters:
        model (nn.Module): The neural network model for which the optimizer is created.

    Returns:
        AdamW: The AdamW optimizer configured with the provided learning rate and epsilon.
    """
    return AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON, no_deprecation_warning=True)
