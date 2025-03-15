# coral counting/assessment using cgras model as proof-of-concept

# try slicing up image using SAHI
from sahi import AutoDetectionModel
# from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = './output'
# test image:
img_dir = '/home/dtsai/Data/victis_datasets/coral_tab_image'
img_list = sorted(Path(img_dir).rglob('*.jpg'))
print(img_list)

img_name = str(img_list[0])

print(img_name)
img = cv.imread(img_name)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.title('original high-res image of a tab')
plt.show()

# following the example: https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb
model_path = '/home/dtsai/Data/victis_datasets/coral_detection_model/20240206_cgras_best.pt'
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path = model_path,
    confidence_threshold=0.1,
    device='cpu'
)

import os

slice_export_dir = os.path.join(out_dir, 'slices')

sliced_results = get_sliced_prediction(
    image=img_name, 
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
    postprocess_type='NMM',
    slice_dir=slice_export_dir,
    slice_export_prefix='slice')


# unfortunately, sahi doesn't have any tools to export visualisations of individual slices (as far as I can see)
# need to do predictions all over again on each set of slices
slices = sorted(Path(slice_export_dir).rglob('*.png'))
print(slices)
print(len(slices))

slice_pred_dir = os.path.join(out_dir, 'slice_predictions')
slice_filenames = []
for i, slice in enumerate(slices):
    # print(slice)
    pred_slice = get_prediction(image=str(slice), 
                                detection_model=detection_model)
    
    output_name =os.path.basename(slice)[:-4]
    print(output_name)
    pred_slice.export_visuals(export_dir=slice_pred_dir, 
                      hide_conf=False,
                      text_size=0.6,
                      rect_th=2,
                      file_name=output_name)
    slice_filenames.append(os.path.join(slice_pred_dir, output_name + '.png'))

# print/show the resultant sliced predictions in an array of images

# Function to extract coordinates from filename
def extract_coords(filename):
    filename = filename[:-4]
    parts = filename.split("_")
    if len(parts) == 5:  # Expected format: slice_x1_y1_x2_y2
        x1, y1, x2, y2 = map(int, parts[1:])
        return x1, y1, x2, y2
    return None
    
# Store slices with their coordinates
slices = []
for filename in slice_filenames:
    coords = extract_coords(os.path.basename(filename))
    if coords:
        slices.append((filename, coords))



import code
code.interact(local=dict(globals(), **locals()))

# Sort slices by (y1, x1) so they appear in correct order

slices.sort(key=lambda x: (x[1][1], x[1][0]))


# Load images in sorted order
sorted_images = [cv.imread(filename) for filename, _ in slices]

# Determine grid size (5x5 for 25 slices)
grid_size = int(np.sqrt(len(sorted_images)))  # Assuming a square layout

print(grid_size)
# Plot slices in correct order
fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
axes = axes.flatten()

for i, img in enumerate(sorted_images):
    axes[i].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axes[i].axis("off")

plt.tight_layout()
plt.show()

import code
code.interact(local=dict(globals(), **locals()))
