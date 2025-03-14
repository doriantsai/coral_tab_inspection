#!/usr/bin/env python3

# display all images

from ImageDisplay import ImageRowDisplay
from pathlib import Path
import cv2 as cv
import os
import matplotlib.pyplot as plt

# import numpy as np

img_dir = '/home/dorian/CodeWSL/tab_assessment/images'
out_dir = '/home/dorian/CodeWSL/tab_assessment/output'
img_list = sorted(Path(img_dir).rglob('*.png'))

figsize = (15,4)
green = (0, 255, 0)
thickness = 1

imgs = []
img_basename = []
MAX_IMG = 100
for i, img_name in enumerate(img_list):
    print(f'{i}: {img_name}')
    if i > MAX_IMG:
        print(f'MAX_IMG reached')
        break
    
    # read in the images:
    img = cv.imread(img_name,cv.IMREAD_COLOR_RGB)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    img_basename.append(os.path.basename(img_name))
    imgs.append(img)

# display
fig_originals = ImageRowDisplay(images=imgs, labels=img_basename, figsize=figsize, title='original images')
fig_originals.show_images()

# now apply shape matching via contours:
img_ref_name = img_list[0] # assume the first image name in the list if the reference image (manually named AA_reference.png)
img_ref = imgs[0]
cont_ref, _ = cv.findContours(img_ref, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

shape_similarity = []
conts = []
imgs_cont = []
for i, img in enumerate(imgs):
    cont, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   
    # only compare the maximum-sized contour: (max area? max perimeter?)
    cont_large = max(cont, key=cv.contourArea)
    
    if cont and cont_ref:
        # compute similarity measure
        similarity = cv.matchShapes(cont_large, cont_ref[0], cv.CONTOURS_MATCH_I1, 0.0)
        
        # prepare label format
        label = "{:.2f}".format(similarity)
        if i == 0:
            # reference image
            label = f'reference: {label}'
        print(f'shape similarity = {label}')
        
        # draw the contour
        img_contour = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(img_contour, cont, -1, green, thickness) #-1 for just contour, no fill
        
        
        conts.append(cont)
        shape_similarity.append(label)
        imgs_cont.append(img_contour)

print(f'shape similarity = {shape_similarity}')
print(shape_similarity)
# make plots
fig_similarity_contour = ImageRowDisplay(images=imgs_cont, labels=shape_similarity, title="similarity based on contours (lower is better)", figsize=figsize)
fig_similarity_contour.show_images()
fig_similarity_contour.save(base_fig_name='shape_similarity.png', dir='/home/dorian/CodeWSL/tab_assessment/output')



# similarity = 0 is perfect similarity
# set a threshold for similarity