################## EXPLANATIONS ##################
# This code was written for downloading the dataset. The 'download_images' function is will be used in ./Creating the Dataset.ipynb
################## EXPLANATIONS ##################
import cv2
import numpy as np
import os
import random
from bing_image_downloader import downloader as d

def download_images(product_list, limit=100, output_dir='./dataset'):
    os.makedirs(output_dir, exist_ok=True)

    for obj in product_list:
        product_dir = os.path.join(output_dir, obj)
        os.makedirs(product_dir, exist_ok=True)

        d.download(obj, limit=limit, adult_filter_off=True)
        
        for i in range(limit):
            image_path = f'./dataset/{obj}/Image_{i}.jpg'
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            

    cv2.destroyAllWindows()
