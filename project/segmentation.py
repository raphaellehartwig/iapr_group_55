import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


### NEUTRAL
def detect_and_display_circles_neutral(img, display=False):
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Apply adaptive thresholding and morphological operations
    img_th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)
    closing = cv2.dilate(closing, kernel, iterations=3)

    # Hough Circle Transform
    circles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=10, minRadius=50, maxRadius=120)

    # Convert the (x, y) coordinates and radius of the circles to integers
    if circles is not None:
        circles = np.uint16(np.around(circles))[0, :]

    # Plot the image with circles
    if display:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if circles is not None:
            #    circles = np.uint16(np.around(circles))[0, :]
            for (x, y, r) in circles:
                plt.gca().add_patch(plt.Circle((x, y), r, color='red', fill=False, linewidth=2))
        plt.title('Detected Circles')
        plt.axis('off')
        plt.show()

    if circles is not None:
        return [(x, y, r) for (x, y, r) in circles]
    else:
        return []


### NOISY
def detect_and_display_circles_noisy(img, display=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Apply Gaussian Blur to the image
    blur = cv2.GaussianBlur(hsv, (11, 11), 2)
    data_h, data_s, data_v = cv2.split(blur)

    # Apply thresholds to isolate the desired features
    img_th = np.zeros(data_s.shape, dtype=np.uint8)
    # (hMin = 0 , sMin = 121, vMin = 0), (hMax = 24 , sMax = 255, vMax = 255)
    # orange coins
    img_th[(data_s > 120) & (data_h < 20)] = 255
    # gray coins
    img_th[(data_s > 82) & (data_h < 23) & (data_s < 170)] = 255
    # for gray bright coins
    img_th[(data_h > 9) & (data_h < 21) & (data_s < 82) & (data_s > 45) & (data_v < 240)] = 255
    img_th[(data_h > 19) & (data_h < 23) & (data_s < 140) & (data_s > 67) & (data_v < 231)] = 255
    # for bright yellow coins
    # (hMin = 19 , sMin = 153, vMin = 210), (hMax = 25 , sMax = 255, vMax = 255)
    img_th[(data_h > 19) & (data_v > 210) & (data_s > 153) & (data_h < 25)] = 255

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel = np.ones((6, 6), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=4)

    # Apply Gaussian Blur to the cleaned image
    #blur = cv2.GaussianBlur(opening, (3, 3), 2)

    # Find contours in the preprocessed image
    circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, dp=1., minDist=80, param1=200, param2=10, minRadius=40,
                               maxRadius=120)

    # Convert the (x, y) coordinates and radius of the circles to integers
    if circles is not None:
        circles = np.uint16(np.around(circles))[0, :]

    # Plot the image with circles
    if display:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if circles is not None:
            #    circles = np.uint16(np.around(circles))[0, :]
            for (x, y, r) in circles:
                plt.gca().add_patch(plt.Circle((x, y), r, color='red', fill=False, linewidth=2))
        plt.title('Detected Circles')
        plt.axis('off')
        plt.show()

    if circles is not None:
        return [(x, y, r) for (x, y, r) in circles]
    else:
        return []


### HAND
def detect_and_display_circles_hand(img, display=False):
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(blurred, 15, 60, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=1, maxLineGap=80)
    mask = np.ones_like(edges) * 255
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), 0, 5)

    edges = cv2.bitwise_and(edges, mask)
    mask_watch = np.ones_like(edges) * 255

    # Mask for the watch
    cv2.rectangle(mask_watch, (0, 850), (1500, 1000), 0, -1)
    edges = cv2.bitwise_and(edges, mask_watch)

    # Apply Hough Circle Transform on the masked image
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=150, param2=22, minRadius=50, maxRadius=100)

    # Convert the (x, y) coordinates and radius of the circles to integers
    if circles is not None:
        circles = np.uint16(np.around(circles))[0, :]

    if display:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if circles is not None:
            #circles = np.uint16(np.around(circles))[0, :]
            for (x, y, r) in circles:
                plt.gca().add_patch(plt.Circle((x, y), r, color='red', fill=False, linewidth=2))
        plt.title('Detected Circles in ')
        plt.axis('off')
        plt.show()

    if circles is not None:
        return [(x, y, r) for (x, y, r) in circles]
    else:
        return []


def crop_coins(image, circles):
    cropped_images = []
    height, width = image.shape[:2]

    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)

        # Calculate the coordinates, ensuring they stay within image bounds
        x_min = max(0, x - r)
        x_max = min(width, x + r)
        y_min = max(0, y - r)
        y_max = min(height, y + r)

        cropped_img = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_img)

    return cropped_images


def detect_circles_classification(img, display=False):
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=60, param2=10, minRadius=50, maxRadius=120)

    if display:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if circles is not None:
            circles = np.uint16(np.around(circles))[0, :]
            for (x, y, r) in circles:
                plt.gca().add_patch(plt.Circle((x, y), r, color='red', fill=False, linewidth=2))
        plt.title('Detected Circles in')
        plt.axis('off')
        plt.show()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        if circles.shape[1] > 1:  # Check if there are multiple circles
            (x, y, r) = circles[0, 0]  # Use the first detected circle
        else:
            (x, y, r) = circles[0, 0]

        return (x, y, r)
    else:
        return (0, 0, 0)


def detect_and_crop_coins(image_type, img=None, img_path=None, display_cropped=False,):
    # Load the image
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    """if img is None:
        print("Error: Image not found at", image_path)
        return None"""

    # Call the appropriate detection function
    if image_type == 'neutral':
        circles = detect_and_display_circles_neutral(img, display=False)
    elif image_type == 'hand':
        circles = detect_and_display_circles_hand(img, display=False)
    elif image_type == 'noisy':
        circles = detect_and_display_circles_noisy(img, display=False)
    else:
        print("Error: Unknown image type")
        return None

    if not circles:
        print("No circles detected")
        return [],[]

    # Crop the detected coins
    cropped_images = crop_coins(img, circles)

    # Display cropped images if required
    if display_cropped:
        for i, cropped_img in enumerate(cropped_images):
            plt.figure(figsize=(4, 4))
            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Cropped Coin {i + 1}')
            plt.axis('off')
            plt.show()

    return cropped_images, circles


def crop_whole_directory(root_path, root_output, img_type, save=True):
    all_images = []
    total_coins = 1
    for root, dirs, files in os.walk(root_path + img_type):
        for i, file in enumerate(files):
            if file.endswith('.JPG'):
                # construct path to the image file
                file_path = os.path.join(root, file)
                if img_type == 'noisy':
                    im = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                else:
                    im = cv2.imread(file_path, cv2.IMREAD_COLOR)
                cropped_img = detect_and_crop_coins(img=np.array(im), image_type=img_type, display_cropped=False)
                all_images.append(cropped_img)
                if save:
                    for j, cropped_img in enumerate(cropped_img):
                        output_path = os.path.join(root_output, f'{total_coins}.png')
                        cv2.imwrite(output_path, cropped_img)
                        total_coins += 1
                        print(f'Saved cropped image to {output_path}')
    return all_images


def process_images_in_directory(directory_path, image_type, display_cropped=True):
    if image_type == 'neutral':
        detect_and_display_circles = detect_and_display_circles_neutral
    elif image_type == 'hand':
        detect_and_display_circles = detect_and_display_circles_hand
    else:
        print("Error: Unknown image type")
        return

    # List all jpg images in the directory
    images = glob.glob(os.path.join(directory_path, '*.JPG'))
    for image_path in images:
        circles = detect_and_display_circles(image_path, display=False)
        if circles:
            print(f"Detected circles in {os.path.basename(image_path)}: {circles}")
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            cropped_images = crop_coins(img, circles)
            if display_cropped:
                for i, cropped_img in enumerate(cropped_images):
                    plt.figure(figsize=(4, 4))
                    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                    plt.title(f'Cropped Coin {i + 1}')
                    plt.axis('off')
                    plt.show()
        else:
            print(f"No circles detected in {os.path.basename(image_path)}")