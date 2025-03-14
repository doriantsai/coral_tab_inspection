# class to display images and specified string in a row

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageRowDisplay:
    def __init__(self, images, labels, title=None, figsize=(12, 4)):
        """
        Initialize the ImageRowDisplay class.
        :param images: List of image arrays (NumPy format, OpenCV loaded images).
        :param labels: List of corresponding text labels.
        :param title: Overall title for the set of images.
        :param figsize: Size of the figure to display the images.
        """
        assert len(images) == len(labels), "Number of images and labels must be the same"
        self.images = images
        self.labels = labels
        self.title = title
        self.figsize = figsize
        self.fig = None

    def show_images(self):
        """Display the images in a row with their corresponding labels below."""
        num_images = len(self.images)
        fig = plt.figure(figsize=self.figsize)
        
        if self.title:
            plt.suptitle(self.title, fontsize=14, fontweight='bold')
        
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            plt.subplot(1, num_images, i + 1)
            
            # Convert BGR (OpenCV format) to RGB for Matplotlib
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            plt.axis('off')
            plt.title(label, fontsize=10)
        
        plt.tight_layout()
        
        self.fig = fig
        plt.show()

    def save(self, base_fig_name, dir=None, dpi=300):
        """ save image """
        if dir is None:
            dir = './' # local directory
        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(dir, base_fig_name)
        self.fig.savefig(filename, dpi=dpi)
        
# Example Usage:
# images = [cv2.imread('image1.jpg'), cv2.imread('image2.jpg')]
# labels = ['First Image', 'Second Image']
# viewer = ImageRowDisplay(images, labels, title='Comparison of Images')
# viewer.show_images()
