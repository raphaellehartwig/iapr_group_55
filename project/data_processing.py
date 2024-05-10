import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset


def preprocess_images(images, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], input_shape=(224, 224)):
    # Preprocess image to match input dim
    preprocess = transforms.Compose([
        transforms.Resize(input_shape),  # Resize to match model's input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=means, std=stds)
        # Normalize pixel values, based on imagenet
    ])

    images_preprocessed = []
    for img in images:
        temp = preprocess(img).unsqueeze(0)
        temp = temp.reshape([3, 224, 224])
        images_preprocessed.append(temp)

    return images_preprocessed


def create_dataset(labels, data_path):
    labels_list = []
    image_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.JPG'):
                # construct path to the image file
                file_path = os.path.join(root, file)
                im = Image.open(file_path)
                image_list.append(im)
                img_id = file[:-4]
                label = labels.loc[img_id].values
                labels_list.append(torch.tensor(label))

    img_tensor_list = preprocess_images(image_list)

    data = TensorDataset(torch.tensor(np.array(labels_list), dtype=torch.float),
                         torch.tensor(np.array(img_tensor_list)))

    return data
