import torch
import time
import os
from tempfile import TemporaryDirectory
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from utils import *
import numpy as np
from segmentation import *
from data_processing import *
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset


class LinearClassifier(nn.Module):
    def __init__(self, input_features=20*256*36, num_classes=3):
        super(LinearClassifier, self).__init__()
        # Define the linear layer
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        output = self.linear(x)
        return output


def create_classifier_from_alexnet(num_outputs):
    model = torchvision.models.alexnet(weights='IMAGENET1K_V1')
    # freeze all parameters (except classification layer)
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer by classifier with 17 outputs (all possible labels)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_outputs)

    return model


def train_model(model, criterion, optimizer, scheduler, data_train, data_val, device, batch_size=4, num_epochs=25,
                verbose=True, phases=['train', 'val'], sampler=None):
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
            if verbose:
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    if sampler is None:
                        dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
                    else:
                        dataloader = DataLoader(data_train, batch_size=batch_size, sampler=sampler)
                    data_length = len(data_train)
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = DataLoader(data_val, batch_size=batch_size)
                    data_length = len(data_val)

                running_loss = 0.0
                running_f1 = 0.0
                running_corrects = 0.0
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
                        preds_batch = torch.argmax(outputs, dim=1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # save validation predictions
                        else:
                            preds.append(((torch.argmax(labels, dim=1)), preds_batch))

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds_batch == torch.argmax(labels, dim=1))
                    running_f1 += f1_score(preds_batch, torch.argmax(labels, dim=1), average='macro')

                epoch_loss = running_loss/data_length
                epoch_accuracy = running_corrects.double()/data_length
                epoch_f1 = running_f1/len(dataloader)

                if verbose:
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
        if 'val' in phases:
            print(f'Best val F1_score: {best_f1:4f} at epoch: {best_epoch}')

        # load best model weights
        #model.load_state_dict(torch.load(best_model_params_path))

    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1, val_f1, preds


def extract_features(images, model):
    features_list = []
    for img in images:
        feature = model(img.unsqueeze(0))
        features_list.append(feature)

    return features_list


def get_all_features(image_list, bg_type, model_features, target_size=20*256*36):
    features_list = []
    for i, img in enumerate(image_list):
        print('Processing image ', i)
        cropped_img = process_image(np.array(img), bg_type, diplay_crop=False) # returns a list of coins images
        cropped_img = preprocess_images(convert_np_to_pil(cropped_img)) # preprocess so that they can be inputted into the model
        if len(cropped_img) != 0:
            features = extract_features(cropped_img, model_features) # extract features for each coin
            features = preprocess_input_features(features, target_size)
        else:
            features = [torch.zeros(target_size)]

        features_list.append(features)

    return features_list


def cross_validate_learning_rate(criterion, optimizer_class, scheduler_class, data, device, num_epochs=11,
                                 batch_size=6, lr_candidates=[0.001, 0.01, 0.1, 1.0], num_folds=5, verbose=True,
                                 model_name='alexnet'):

    best_lr = lr_candidates[0]
    best_f1 = 0
    results = []

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for lr in lr_candidates:
        fold_f1_scores = []
        print(f'Computing F1 score for learning-rate {lr}')
        for train_idx, val_idx in kf.split(data):
            train_subset = Subset(data, train_idx)
            val_subset = Subset(data, val_idx)

            if model_name == 'alexnet':
                model = torchvision.models.alexnet(weights='IMAGENET1K_V1')  # Create a new instance of the model
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, 3)
                params = model.classifier.parameters()
            if model_name == 'resnet':
                model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 8)
                params = model.fc.parameters()

            model = model.to(device)
            optimizer = optimizer_class(params, lr=lr)
            scheduler = scheduler_class(optimizer, step_size=7, gamma=0.1)

            _, _, _, _, _, _, val_f1, _ = train_model(model, criterion, optimizer, scheduler, train_subset,
                                                      val_subset, device, num_epochs=num_epochs, batch_size=batch_size,
                                                      verbose=False)
            fold_f1_scores.append(torch.max(torch.tensor(val_f1)))  # Use the max F1

        avg_f1 = sum(fold_f1_scores) / num_folds
        results.append((lr, avg_f1))
        if verbose:
            print(f'Learning rate: {lr}, Average F1-score: {avg_f1}')

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_lr = lr

    return best_lr, results
