import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset
import pandas as pd
import torch.nn.functional as F


def get_3class_labels(data_path='data/train_labels.csv'):
    labels = pd.read_csv(data_path)
    labels = labels.set_index('id')
    labels['n_CHF'] = labels[['5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF']].sum(axis=1)
    labels['n_EUR'] = labels[['2EUR', '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR', '0.01EUR']].sum(axis=1)
    labels = labels[['n_CHF', 'n_EUR', 'OOD']]

    return labels


def preprocess_images(images, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], input_shape=(224, 224)):
    # Preprocess image to match input dim
    preprocess = transforms.Compose([
        transforms.Resize(input_shape),  # Resize to match model's input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=means, std=stds) # Normalize pixel values, based on imagenet
    ])

    images_preprocessed = []
    for img in images:
        temp = preprocess(img).unsqueeze(0)
        temp = temp.reshape([3, 224, 224])
        images_preprocessed.append(temp)

    return images_preprocessed


def load_data(labels, data_path):
    labels_list = []
    image_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.JPG'):
                # construct path to the image file
                file_path = os.path.join(root, file)
                #im = Image.open(file_path)
                im = cv2.imread(file_path, cv2.IMREAD_COLOR)
                image_list.append(im)
                img_id = file[:-4]
                label = labels.loc[img_id].values
                labels_list.append(torch.tensor(label))

    return image_list, labels_list


def preprocess_input_features(input_features, target_size=20*256*36): # target size = n_coins * features dim
    """
    Preprocess input features by padding or cropping them to the target size.

    Args:
        input_features (torch.Tensor): Input features tensor.
        target_size (int): Target size to pad or crop the input features.

    Returns:
        torch.Tensor: Preprocessed input features tensor.
    """
    input_features = torch.tensor(np.array(input_features))
    #input_features = input_features.reshape((1, input_features.shape[0]*input_features.shape[1]*input_features.shape[2]))
    input_features = input_features.reshape(input_features.shape[0] * input_features.shape[1] * input_features.shape[2])
    current_size = input_features.shape[-1]

    # Pad or crop the input features to the target size
    if current_size < target_size:
        # Pad with zeros to the right
        pad_amount = target_size - current_size
        input_features = F.pad(input_features, (0, pad_amount))
    elif current_size > target_size:
        # Crop from the right
        input_features = input_features[:target_size]

    return input_features #.reshape((1, target_size))
