#!/usr/bin/env/ python3

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class coral_tab_detector:
    def __init__(self,
                 save_dir,
                 model_path,
                 conf_thresh=0.3,
                 device='cpu'):
        # self.img_name = img_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.detection_model = self.load_model(self.model_path, device=device)

        self.slice_height=640
        self.slice_width=640
        self.overlap_height_ratio=0.3
        self.overlap_width_ratio=0.3
        
        self.slice_export_dir = os.path.join(self.save_dir, 'slices')
        self.slice_export_prefix = 'slice'
        self.slice_pred_export_dir = os.path.join(self.save_dir, 'slice_predictions')

        self.text_size_slice=0.6
        self.rect_th_slice=2

        self.text_size_overall=0.6
        self.rect_th_overall=2
        


    def load_model(self, model_path, device='cpu'):
        return AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path = model_path,
            confidence_threshold=self.conf_thresh,
            device=device
        )



    # Function to extract coordinates from filename
    def extract_coords(self, filename):
        parts = filename.split("_")
        if len(parts) == 5:  # Expected format: slice_x1_y1_x2_y2
            x1, y1, x2, y2 = map(int, parts[1:])
            return x1, y1, x2, y2
        return None

    def predict_overall(self, img_name, save_file):
        sliced_results = get_sliced_prediction(
            image=img_name,
            detection_model=self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            postprocess_type='NMM',
            slice_dir=self.slice_export_dir,
            slice_export_prefix=self.slice_export_prefix
        )

        sliced_results.export_visuals(export_dir=self.save_dir, 
                      hide_conf=False,
                      text_size=self.text_size_overall,
                      rect_th=self.rect_th_overall,
                      file_name=save_file)
        
        return sliced_results
    
    def predict_for_slices(self, img_name, save_predicted_slice_array_name):
        # unfortunately, sahi doesn't have any tools to export visualisations of individual slices (as far as I can see)
        # need to do predictions all over again on each set of slices
        sliced_results = get_sliced_prediction(
            image=img_name,
            detection_model=self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            postprocess_type='NMM',
            slice_dir=self.slice_export_dir,
            slice_export_prefix=self.slice_export_prefix
        )

        # NOTE an quick/dirty write to disk, then read from disk operation - ideally, should do all within memory
        slices = sorted(Path(self.slice_export_dir).rglob('*.png'))
        
        slice_coords = []
        for slice in slices:
            filename = os.path.basename(slice)[:-4]
            pred_slice = get_prediction(image=str(slice),
                                        detection_model=self.detection_model)
            
            pred_slice.export_visuals(export_dir=self.slice_pred_export_dir,
                                      hide_conf=False,
                                      text_size=self.text_size_slice,
                                      rect_th=self.rect_th_slice,
                                      file_name=filename)
            coords = self.extract_coords(filename)
            if coords:
                slice_coords.append((filename, coords))


        # Sort slices by (y1, x1) so they appear in correct order
        slice_coords.sort(key=lambda x: (x[1][1], x[1][0]))

        # Load images in sorted order
        sorted_images = [cv.imread(os.path.join(self.slice_pred_export_dir, filename+'.png')) for filename, _ in slice_coords]

        # Determine grid size (5x5 for 25 slices)
        grid_size = int(np.sqrt(len(sorted_images)))  # Assuming a square layout

        # plot slices with predictions in an array
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten()

        for i, img in enumerate(sorted_images):
            axes[i].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axes[i].axis("off")

        plt.tight_layout()
        fig.savefig(save_predicted_slice_array_name, dpi=300)
        # plt.show()
        return sliced_results


if __name__ == '__main__':

    
    img_dir = '/home/dtsai/Data/victis_datasets/coral_tab_simulated_images'
    img_list = sorted(Path(img_dir).rglob('*.jpg'))
    img_list = [str(path) for path in img_list]
    print(img_list)
    idx=1
    print(f'IMAGE SELECTED: {idx}: {img_list[idx]}')

    model_path = '/home/dtsai/Data/victis_datasets/coral_detection_model/20240206_cgras_best.pt'
    save_dir = os.path.join('output', os.path.basename(img_list[idx])[:-4])

    detector = coral_tab_detector(save_dir=save_dir,
                                  model_path=model_path,
                                  device='cuda:0')
    
    detector.predict_for_slices(img_name=img_list[idx],
                                save_predicted_slice_array_name=os.path.join(save_dir, os.path.basename(img_list[idx])[:-4]+'_slice_array'))

    sliced_results = detector.predict_overall(img_name=img_list[idx],
                                              save_file=os.path.basename(img_list[idx])[:-4]+'_combined')

    import code
    code.interact(local=dict(globals(), **locals()))