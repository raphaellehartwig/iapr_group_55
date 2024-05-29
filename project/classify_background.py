# Import main packages
import cv2
import numpy as np


def classify_background(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    threshold_noisy = ((image[:,:,0] > 60) & (image[:,:,0] < 130))
    threshold_hand = (hsv_image[:,:,0] > 125)

    if (np.sum(threshold_noisy)) > 1000000:
        background = "noisy"
        #print(background)
        return background
       
    elif np.sum(threshold_hand) > 500000:
        background = "hand"
        #print(background)
        return background
    
    else:
        background = "neutral"
        #print(background)
        return background


def classify_images_in_folders(base_folder):
    folders = ['Neutral', 'Noisy', 'Hand']
    correct_classifications = 0
    total_images = 0
    
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        actual_label = folder
        
        for filename in os.listdir(folder_path):
            print(f"Processing {filename}")
            img_path = os.path.join(folder_path, filename)
            test_img = cv2.imread(img_path)
            img_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

            predicted_label = classify_background(img_test)
            if predicted_label == actual_label:
                correct_classifications += 1
            total_images += 1

    classification_rate = correct_classifications / total_images if total_images > 0 else 0
    print(f"Classification rate: {classification_rate * 100:.2f}%")