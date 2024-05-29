import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from segmentation import *
from classify_background import *


### CLASSIFY BACKGROUND
def classify_area(area):
    # 5
    if area > 25000:
        return 0
    # 2
    elif area > 17000:
        return 1
    # 1
    elif area > 12000:
        return 2
    # 0.2
    elif area > 10800:
        return 4
    # 0.1
    elif area > 10000:
        return 5
    # 0.5
    elif area > 9000:
        return 3
    # 0.05
    else:
        return 6
    

def classify_area_b(area, B):
    # 0.05
    if B < 97 and area < 9000:
        return 6
    # 5
    if area > 24000:
        return 0
    # 2
    elif area > 17000:
        return 1
    # 1
    elif area > 12600:
        return 2
    # 0.2
    elif area > 10800:
        return 4
    # 0.1
    elif area > 10000:
        return 5
    # 0.5
    else:
        return 3
    

def circle_area(circle):
    (x, y, r) = circle
    area = np.pi * r ** 2

    return area


def get_blue(image):
    # Calculate the center of the image
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Calculate the coordinates for the 50x50 square
    top_left_x = max(center_x - 25, 0)
    top_left_y = max(center_y - 25, 0)
    bottom_right_x = min(center_x + 25, width)
    bottom_right_y = min(center_y + 25, height)

    # Extract the 50x50 region from the middle of the image
    square = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    data_r, data_g, data_b = cv2.split(rgb)

    return data_b.mean()


### CLASSIFY CHF/EUR/OOD
def load_model1(model_path1):
    model1 = torchvision.models.alexnet(weights='IMAGENET1K_V1')

    num_ftrs = model1.classifier[-1].in_features
    model1.classifier[-1] = nn.Linear(num_ftrs, 3)

    model1.load_state_dict(torch.load(model_path1))

    return model1


### CLASSIFY EUR SUBCLASSES
def load_model2(model_path2):
    model2 = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    num_ftrs = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs, 8)

    model2.load_state_dict(torch.load(model_path2))

    return model2


def transform(coin):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match model's input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize pixel values, based on imagenet
    ])

    transformed_coin = transform(coin)
    return transformed_coin


def load_images_and_backgrounds(folder, downsampled_folder, downsampled=True):
    images = []
    ids = []
    backgrounds = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.JPG'):
                print(f"Processing file {file}")
                # Construct path to the image file
                file_path = os.path.join(root, file)
                img_id = file[:-4]
                im_original = cv2.imread(file_path, cv2.IMREAD_COLOR)

                # Classify background
                background = classify_background(im_original)
                downsampled_path = os.path.join(downsampled_folder, file)
                if background == 'noisy':
                    im_reduced = cv2.imread(downsampled_path, cv2.IMREAD_UNCHANGED)

                else:
                    im_reduced = cv2.imread(downsampled_path, cv2.IMREAD_COLOR)
                images.append(im_reduced)
                    
                ids.append(img_id)
                backgrounds.append(background)
                #print(background)

    return images, ids, backgrounds


def predict(images, ids, backgrounds, model_path1, model_path2):
    predictions = {}
    subclasses = {}

    model1 = load_model1(model_path1)
    model2 = load_model2(model_path2)

    model1.eval()
    model2.eval()

    for i, (background, img) in enumerate(zip(backgrounds, images)):
        print(ids[i])
        cropped_images, circles = detect_and_crop_coins(background, np.array(img))

        pred = np.zeros(3)
        subclass = np.zeros(16)
        outputs = []

        if len(cropped_images) != 0:
            for coin, circle in zip(cropped_images, circles):
                coin_np = coin
                #print(coin_np)
                new_circle = detect_circles_classification(coin)
                coin = transform(coin)
                coin = coin.unsqueeze(0)

                with torch.no_grad():
                    outputs = model1(coin)
                output = torch.argmax(outputs) # index 0 -> CHF, index 1 -> EUR, index 2 -> OOD
                
                if output == 0:
                    area = circle_area(new_circle)
                    data_b = get_blue(coin_np)
                    label = classify_area_b(area, data_b)
                    #label = classify_area(area)
                
                if output == 1:
                    with torch.no_grad():
                        outputs2 = model2(coin) # index 0 -> 0.01EUR, ..., index 7 -> 2EUR
                    inverted_label = 7 - torch.argmax(outputs2).item() # index 0 -> 2EUR, ..., index 7 -> 0.01EUR
                    label = 7 + inverted_label # index 7 -> 2EUR, ..., index 14 -> 0.01EUR

                if output == 2:
                    label = 15
                
                subclass[label] += 1 
                subclasses[ids[i]] = subclass
                pred[output] += 1 # add the coin to the image prediction

        else:
            print(f'no coins detected in image {i}')

        predictions[ids[i]] = pred
        
    return predictions, subclasses


def generate_csv_file(subclasses, csv_path):
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(subclasses, orient='index')

    # Define the column names
    columns = ['id', '5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF', 
            '2EUR', '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR', '0.01EUR', 'OOD']

    # Insert the index as a column
    df.reset_index(inplace=True)
    df.columns = columns
    df.to_csv(csv_path, index=False)
