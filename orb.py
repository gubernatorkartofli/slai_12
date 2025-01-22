import cv2 as cv
import matplotlib.pyplot as plt
import os

# ORB (Oriented FAST and Rotated BRIEF) feature detection and display
def ORB():
    # Define the image path
    root = os.getcwd()
    imgPath = os.path.join(root, 'dwayne.png')
    # Read the image in grayscale
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    # Check if the image loaded successfully
    if imgGray is None:
        print("Error loading image. Please check the file path.")
        return
    # Initialize the ORB detector
    orb = cv.ORB_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(imgGray, None)
    # Draw keypoints on the image
    imgWithKeypoints = cv.drawKeypoints(imgGray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the result
    plt.figure(figsize=(100, 100))
    plt.imshow(imgWithKeypoints, cmap='gray')
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    ORB()
