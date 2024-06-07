import torch
from keras import backend as K
from transformers import AdamW



#### Hyperparameters for training transformers models ####
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-05
EPSILON = 1e-8
WEIGHT_DECAY = 0.001



def f1(y_true, y_pred):
    """
    Computes the F1 score, the harmonic mean of precision and recall, for the given true and predicted labels.

    Parameters:
        y_true (Tensor): The ground truth binary labels.
        y_pred (Tensor): The predicted binary labels.

    Returns:
        Tensor: The F1 score.
    """
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



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
