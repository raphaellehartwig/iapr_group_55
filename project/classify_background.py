# Import main packages
import cv2
import numpy as np


def classify_background(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    threshold_noisy = ((image[:,:,0] > 60) & (image[:,:,0] < 130))
    threshold_hand = (hsv_image[:,:,0] > 125)

    if (np.sum(threshold_noisy)) > 1000000:
        background = "Noisy"
        print(background)
        return background
       
    elif np.sum(threshold_hand) > 500000:
        background = "Hand"
        print(background)
        return background
    
    else:
        background = "Neutral"
        print(background)
        return background
    