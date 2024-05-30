# Import main packages
import cv2
import numpy as np
from PIL import Image
import os
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

def classify_background(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    threshold_noisy = ((image[:,:,0] > 60) & (image[:,:,0] < 130))
    threshold_hand = (hsv_image[:,:,0] > 125)

    if (np.sum(threshold_noisy)) > 1000000:
        background = "noisy"
        return background
       
    elif np.sum(threshold_hand) > 500000:
        background = "hand"
        return background
    
    else:
        background = "neutral"
        return background

def classify_images_in_folders(base_folder):
    folders = ['neutral', 'noisy', 'hand']
    correct_classifications = 0
    total_images = 0
    
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        actual_label = folder
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            test_img = cv2.imread(img_path)
            #img_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

            predicted_label = classify_background(test_img)
            if predicted_label == actual_label:
                correct_classifications += 1
            total_images += 1

    classification_rate = correct_classifications / total_images if total_images > 0 else 0
    print(f"Classification rate: {classification_rate * 100:.2f}%")

def get_mean_hsv(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array
    image_rgb = np.array(image)

    # Convert RGB to HSV channels
    image_hsv = rgb2hsv(image_rgb)
    
    # Calculate the mean for each channel
    mean_h = np.mean(image_hsv[:, :, 0])
    mean_s = np.mean(image_hsv[:, :, 1])
    mean_v = np.mean(image_hsv[:, :, 2])
    
    return mean_h, mean_s, mean_v

def get_mean_rgb(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array
    image_rgb = np.array(image)
    
    # Calculate the mean for each channel
    mean_r = np.mean(image_rgb[:, :, 0])
    mean_g = np.mean(image_rgb[:, :, 1])
    mean_b = np.mean(image_rgb[:, :, 2])
    
    return mean_r, mean_g, mean_b

def compute_mean_hsv(folder_path):
    mean_hs = []
    mean_ss = []
    mean_vs = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        mean_h, mean_s, mean_v = get_mean_hsv(image_path)
        mean_hs.append(mean_h)
        mean_ss.append(mean_s)
        mean_vs.append(mean_v)
    
    # Compute the overall mean for each channel
    overall_mean_h = np.mean(mean_hs)
    overall_mean_s = np.mean(mean_ss)
    overall_mean_v = np.mean(mean_vs)

    # Compute the min and max values for each mean
    min_mean_h = np.min(mean_hs)
    min_mean_s = np.min(mean_ss)
    min_mean_v = np.min(mean_vs)
    
    max_mean_h = np.max(mean_hs)
    max_mean_s = np.max(mean_ss)
    max_mean_v = np.max(mean_vs)

    return (overall_mean_h, overall_mean_s, overall_mean_v,
            min_mean_h, min_mean_s, min_mean_v,
            max_mean_h, max_mean_s, max_mean_v)


def compute_mean_rgb(folder_path):
    mean_rs = []
    mean_gs = []
    mean_bs = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        mean_r, mean_g, mean_b = get_mean_rgb(image_path)
        mean_rs.append(mean_r)
        mean_gs.append(mean_g)
        mean_bs.append(mean_b)
    
    # Compute the overall mean for each channel
    overall_mean_r = np.mean(mean_rs)
    overall_mean_g = np.mean(mean_gs)
    overall_mean_b = np.mean(mean_bs)

    # Compute the min and max values for each mean
    min_mean_r = np.min(mean_rs)
    min_mean_g = np.min(mean_gs)
    min_mean_b = np.min(mean_bs)
    
    max_mean_r = np.max(mean_rs)
    max_mean_g = np.max(mean_gs)
    max_mean_b = np.max(mean_bs)

    return (overall_mean_r, overall_mean_g, overall_mean_b,
+            min_mean_r, min_mean_g, min_mean_b,
            max_mean_r, max_mean_g, max_mean_b)

def compute_and_plot_rgb_histogram(image, bins=256):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Compute and plot the histogram for each RGB channel
    plt.figure(figsize=(15, 5))
    colors = ('red', 'green', 'blue')
    channels = ('R', 'G', 'B')

    for i, color in enumerate(colors):
        hist, bins = np.histogram(image_array[..., i], bins=bins, range=(0, 256))
        plt.subplot(1, 3, i + 1)
        plt.plot(bins[:-1], hist, color=color)
        plt.title(f'{channels[i]} channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
    
    plt.suptitle('RGB Color Histogram')
    plt.show()

def compute_and_plot_b_histogram(image, ax, bins=256):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Compute and plot the histogram for the blue channel
    channel_index = 2  # Index for the blue channel
    color = 'blue'
    channel_name = 'B'

    hist, bins = np.histogram(image_array[..., channel_index], bins=bins, range=(0, 256))
    ax.plot(bins[:-1], hist, color=color)
    ax.set_title(f'{channel_name} channel')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, 256])

    # Add vertical lines at 60 and 130 for the blue channel plot
    ax.axvline(x=60, color='red', linestyle='--')
    ax.axvline(x=130, color='red', linestyle='--')

def compute_and_plot_hsv_histogram(image, bins=256):
    # Convert the image to HSV color space
    hsv_image = image.convert('HSV')
    # Convert image to numpy array
    image_array = np.array(hsv_image)
    
    # Compute and plot the histogram for each HSV channel
    plt.figure(figsize=(15, 5))
    colors = ('orange', 'green', 'blue')
    channels = ('H', 'S', 'V')

    for i, color in enumerate(colors):
        hist, bins = np.histogram(image_array[..., i], bins=bins, range=(0, 256))
        plt.subplot(1, 3, i + 1)
        plt.plot(bins[:-1], hist, color=color)
        plt.title(f'{channels[i]} channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
    
    plt.suptitle('HSV Color Histogram')
    plt.show()
    
def compute_and_plot_v_histogram(image, ax, bins=256):
    # Convert the image to HSV color space
    hsv_image = image.convert('HSV')
    # Convert image to numpy array
    image_array = np.array(hsv_image)
    
    # Compute and plot the histogram for the V channel
    channel_index = 2  # Index for the V channel
    color = 'blue'
    channel_name = 'V'

    hist, bins = np.histogram(image_array[..., channel_index], bins=bins, range=(0, 256))
    ax.plot(bins[:-1], hist, color=color)
    ax.set_title(f'{channel_name} channel')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, 256])

    # Add vertical line at 125 for the V channel plot
    ax.axvline(x=125, color='red', linestyle='--')

def plot_mean_min_max_HSV(mean_std_values):
    # Prepare data for plotting
    categories = list(mean_std_values.keys())
    colors = ['red', 'green', 'blue']

    # Create scatter plots for Mean, Std, Min, and Max H, S, and V
    plt.figure(figsize=(14, 10))

    for i, channel in enumerate(['H', 'S', 'V']):
        plt.subplot(3, 1, i+1)
        
        # Plot the means
        plt.scatter(categories, [mean_std_values[cat][channel][0] for cat in categories], color=colors[i], marker='o', label='Mean')
        
        # Add error bars for the standard deviations
        #plt.errorbar(categories, [mean_std_values[cat][channel][0] for cat in categories], yerr=[mean_std_values[cat][channel][1] for cat in categories], fmt='none', ecolor='black', capsize=5, label='Std')
        
        # Plot the min and max values
        plt.scatter(categories, [mean_std_values[cat][channel][1] for cat in categories], color='orange', marker='x', label='Min')
        plt.scatter(categories, [mean_std_values[cat][channel][2] for cat in categories], color='purple', marker='^', label='Max')
        
        # Annotate the standard deviations
        for j, cat in enumerate(categories):
            plt.annotate(f'{mean_std_values[cat][channel][1]:.2f}', (cat, mean_std_values[cat][channel][0] + mean_std_values[cat][channel][1]), textcoords="offset points", xytext=(0,10), ha='center', color='black')
        
        # Add a horizontal line at y=190 for the H channel plot
        if channel == 'H':
            plt.axhline(y=0.15, color='black', linestyle='--', label='Threshold')
        # Add a horizontal line at y=190 for the S channel plot
        elif channel == 'S':
            plt.axhline(y=0.125, color='black', linestyle='--', label='Threshold')
        # Add a horizontal line at y=180 for the V channel plot
        elif channel == 'V':
            plt.axhline(y=0.77, color='black', linestyle='--', label='Threshold')
        
        plt.title(f'Mean {channel} Values')
        plt.xlabel('Category')
        plt.ylabel(f'{channel}')
        plt.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_mean_min_max_RGB(mean_std_values):
    # Prepare data for plotting
    categories = list(mean_std_values.keys())
    colors = ['red', 'green', 'blue']

    # Create scatter plots for Mean, Std, Min, and Max Red, Green, and Blue
    plt.figure(figsize=(14, 10))

    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(3, 1, i+1)
        
        # Plot the means
        plt.scatter(categories, [mean_std_values[cat][channel][0] for cat in categories], color=colors[i], marker='o', label='Mean')
        
        # Add error bars for the standard deviations
        #plt.errorbar(categories, [mean_std_values[cat][channel][0] for cat in categories], yerr=[mean_std_values[cat][channel][1] for cat in categories], fmt='none', ecolor=colors[i], capsize=5, label='Std')
        
        # Plot the min and max values
        plt.scatter(categories, [mean_std_values[cat][channel][1] for cat in categories], color=colors[i], marker='x', label='Min')
        plt.scatter(categories, [mean_std_values[cat][channel][2] for cat in categories], color=colors[i], marker='^', label='Max')
        
        # Annotate the standard deviations
        for j, cat in enumerate(categories):
            plt.annotate(f'{mean_std_values[cat][channel][1]:.2f}', (cat, mean_std_values[cat][channel][0] + mean_std_values[cat][channel][1]), textcoords="offset points", xytext=(0,10), ha='center', color=colors[i])
        
        # Add a horizontal line at y=190 for the Red channel plot
        if channel == 'Red':
            plt.axhline(y=185, color='black', linestyle='--', label='Threshold')
        # Add a horizontal line at y=190 for the Green channel plot
        elif channel == 'Green':
            plt.axhline(y=190, color='black', linestyle='--', label='Threshold')
        # Add a horizontal line at y=180 for the Blue channel plot
        elif channel == 'Blue':
            plt.axhline(y=180, color='black', linestyle='--', label='Threshold')
        
        plt.title(f'Mean {channel} Values')
        plt.xlabel('Category')
        plt.ylabel(f'{channel}')
        #plt.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()