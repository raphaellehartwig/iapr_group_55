import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
import cv2


def downsample_images(original_path, save_path, size=(1500, 1000)):
    """
    Downsamples all images in original_path and saves them in save_path
    :param original_path: where to load the images from
    :param save_path: where to save the images
    :param size: size to downsample the image to
    :return:
    """
    for root, dirs, files in os.walk(original_path):
        for file in files:
            if file.endswith('.JPG'):
                # construct path to the image file
                file_path = os.path.join(root, file)
                im = Image.open(file_path)
                # downsample
                downsampled_image = im.resize(size)
                # save image
                downsampled_image_save_path = os.path.join(save_path, file)
                downsampled_image.save(downsampled_image_save_path)


def plot_metric(train_metrics, val_metrics, title, ylabel):
    """
    Plot train and val metrics on the same graph.
    """
    plt.plot(train_metrics, label='train')
    plt.plot(val_metrics, label='validation')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def compute_f1(outputs, labels):
    """
    Compute the f1 score according to the formula on Kaggle.
    :param outputs: predicted labels
    :param labels: ground truth labels
    :return: f1-score
    """
    preds = np.array(outputs)
    labels = np.array(labels)
    score = 0
    for i, true_label in enumerate(labels):
        TP = np.minimum(true_label, preds[i]).sum()
        FPN = np.abs(true_label-preds[i]).sum()
        if 2*TP + FPN != 0:
            score += 2*TP/(2*TP + FPN)

    return score/len(labels)


def convert_np_to_pil(image_array):
    """
    Convert a numpy array to a pil image.
    """
    pil_images = []
    for img in image_array:
        # Convert numpy array (height, width, channels) to PIL image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # Assuming the input numpy array has values in [0, 1]
        pil_images.append(pil_img)
    return pil_images

def calculate_circle_properties(circles, images):
    """
    Given a list of images and a corresponding list of circles, computes:
    - the perimeter and the area of the circle
    - the mean H, S, V, R, G and B values of the center 50x50 pixels
    """
    if circles is None:
        return []

    properties = []

    for i, (x, y, r) in enumerate(circles):
        perimeter = 2 * np.pi * r
        area = np.pi * r ** 2

        # Calculate the center of the image
        height, width = images[i].shape[:2]
        center_x, center_y = width // 2, height // 2

        # Calculate the coordinates for the 50x50 square
        top_left_x = max(center_x - 25, 0)
        top_left_y = max(center_y - 25, 0)
        bottom_right_x = min(center_x + 25, width)
        bottom_right_y = min(center_y + 25, height)

        # Extract the 50x50 region from the middle of the image
        square = images[i][top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        data_h, data_s, data_v = cv2.split(hsv)
        data_r, data_g, data_b = cv2.split(rgb)

        properties.append((perimeter, area, data_h.mean(), data_s.mean(), data_v.mean(),
                           data_r.mean(), data_g.mean(), data_b.mean()))


    return properties
