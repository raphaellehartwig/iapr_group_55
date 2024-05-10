import torch
import time
import os
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def compute_pred(outputs):
    outputs = outputs.detach()
    outputs_rounded = torch.round(outputs)
    outputs_rounded[outputs_rounded < 0] = 0

    return outputs_rounded


def compute_accuracy(outputs, labels):
    preds = compute_pred(outputs)
    n_correct = (np.array(preds) == np.array(labels.tolist())).sum()
    accuracy = n_correct/labels.numel()
    return accuracy


def compute_f1(outputs, labels):
    preds = compute_pred(outputs)
    preds = np.array(preds)
    labels = np.array(labels)
    score = 0
    for i, true_label in enumerate(labels):
        TP = np.minimum(true_label, preds[i]).sum()
        FPN = np.abs(true_label-preds[i]).sum()
        score += 2*TP/(2*TP + FPN)

    return score/len(labels)


def train_model(model, criterion, optimizer, scheduler, data_train, data_val, device, batch_size=4, num_epochs=25):
    since = time.time()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1 = []
    val_f1 = []
    preds = []
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_f1 = 0
        best_epoch = num_epochs

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
                    data_length = len(data_train)
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = DataLoader(data_val, batch_size=batch_size)
                    data_length = len(data_val)

                running_loss = 0.0
                running_accuracy = 0.0
                running_f1 = 0.0
                # Iterate over data.
                for labels, inputs in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # save validation predictions
                        else:
                            preds.append((labels, compute_pred(outputs)))

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_accuracy += compute_accuracy(outputs, labels)
                    running_f1 += compute_f1(outputs, labels)

                epoch_loss = running_loss/data_length
                epoch_accuracy = running_accuracy/len(dataloader)
                epoch_f1 = running_f1/len(dataloader)
                print(f'{phase} Loss: {epoch_loss:.4f}')
                print(f'{phase} Accuracy: {epoch_accuracy:.4f}')
                print(f'{phase} F1-score: {epoch_f1:.4f}')

                if phase == 'train':
                    scheduler.step()
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_accuracy)
                    train_f1.append(epoch_f1)
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_accuracy)
                    val_f1.append(epoch_f1)

                # deep copy the model
                if phase == 'val' and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val F1_score: {best_f1:4f} at epoch: {best_epoch}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1, val_f1, preds