import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq

from src.config_train import *

def train_model(training_loader, model, optimizer, loss_fn, device):
    """
    Train the model for one epoch.

    Args:
    training_loader (DataLoader): DataLoader for training data.
    model (torch.nn.Module): The model to be trained.
    optimizer (torch.optim.Optimizer): Optimizer for the model.
    loss_fn (callable): Loss function.
    device (torch.device): The device tensors will be allocated to.

    Returns:
    tuple: A tuple containing:
        - model (torch.nn.Module): The trained model.
        - float: The accuracy of the model on the training set.
        - float: The average loss over the training set.
    """

    print("Training:")
    losses = []
    correct_predictions = 0
    num_samples = 0

    model.train()

    loop = tq.tqdm(enumerate(training_loader), total=len(training_loader),
                   leave=True, colour='steelblue')

    for batch_idx, data in loop:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        emotions = data['emotion'].to(device, dtype = torch.float)

        # forward
        outputs = model(ids, mask)
        loss = loss_fn(outputs, emotions)
        losses.append(loss.item())

        # training accuracy
        _, preds = torch.max(outputs, dim=1) # batch dim
        _, emotion = torch.max(emotions, dim=1)  # batch dim
        num_samples += len(emotion)
        correct_predictions += torch.sum(preds==emotion)

        # backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # grad descent step
        optimizer.step()


        # Update progress bar
        loop.set_description('Training')
        loop.set_postfix(loss=loss.item(), acc=float(correct_predictions) / num_samples)

    return model, float(correct_predictions) / num_samples, np.mean(losses)


def eval_model(validation_loader, model, loss_fn, device):
    """
    Evaluate the model on the validation set.

    Args:
    validation_loader (DataLoader): DataLoader for validation data.
    model (torch.nn.Module): The model to be evaluated.
    loss_fn (callable): Loss function.
    device (torch.device): The device tensors will be allocated to.

    Returns:
    tuple: A tuple containing:
        - float: The accuracy of the model on the validation set.
        - float: The average loss over the validation set.
    """

    print("Evaluating:")
    losses = []
    correct_predictions = 0
    num_samples = 0

    model.eval()
    loop = tq.tqdm(enumerate(validation_loader), total=len(validation_loader),
                   leave=True, colour='#8B0000')

    with torch.no_grad():
        for batch_idx, data in loop:
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            emotions = data['emotion'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            loss = loss_fn(outputs, emotions)
            losses.append(loss.item())

            _, preds = torch.max(outputs, dim=1)
            _, emotion = torch.max(emotions, dim=1)
            num_samples += len(emotion)
            correct_predictions += torch.sum(preds == emotion)

            # Update progress bar
            loop.set_description('Evaluating')
            loop.set_postfix(loss=loss.item(), acc=float(correct_predictions) / num_samples)

    return float(correct_predictions) / num_samples, np.mean(losses)


def get_predictions(model, data_loader, device):
    """
    Obtain predictions from a model based on the input from a data loader.

    This function processes batches of sentences from the data loader, performs
    predictions using the provided model, and gathers the outputs including sentences,
    predictions, prediction probabilities, and emotion values.

    Args:
        model (torch.nn.Module): The model to use for predictions, expected to be in evaluation mode.
        data_loader (DataLoader): A PyTorch DataLoader containing the dataset for prediction.

    Returns:
        tuple: A tuple containing four lists:
            - sentences (list[str]): The original sentences from the data loader.
            - predictions (list[int]): The predicted class indices for each sentence.
            - prediction_probs (list[list[float]]): The probabilities for each class for each sentence.
            - emotion_values (list[int]): The emotion values extracted from the dataset.
    """
    model.eval()

    sentences = []
    predictions = []
    prediction_probs = []
    emotion_values = []

    with torch.no_grad():
        for data in data_loader:
            sentence = data["sentence"]
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            emotions = data["emotion"].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            _, preds = torch.max(outputs, dim=1)
            _, emotion = torch.max(emotions, dim=1)

            sentences.extend(sentence)
            predictions.extend(preds.tolist())  # Convert tensor to list
            prediction_probs.extend(outputs.tolist())  # Convert tensor to list
            emotion_values.extend(emotions.argmax(dim=1).tolist())  # Convert tensor to list

    return sentences, predictions, prediction_probs, emotion_values