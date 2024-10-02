################## EXPLANATIONS ##################
# This code was written for rotaing and augmenting images. The 'apply_augmentation_to_images'function will be used in ./Creating the Dataset.ipynb
################## EXPLANATIONS ##################
import cv2
import os
import random

def apply_augmentation_to_images(product_list, base_dir='./dataset', num_versions=4):

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    for obj in product_list:
        product_dir = os.path.join(base_dir, obj)
        
        if not os.path.exists(product_dir):
            print(f"No images found for {obj} in {product_dir}")
            continue

        for img_name in os.listdir(product_dir):
            img_path = os.path.join(product_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load {img_name}")
                continue

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)


            for j in range(num_versions):
                angle = random.uniform(-45, 45)  
                scale = random.uniform(0.5, 2.0) 
                M = cv2.getRotationMatrix2D(center, angle, scale)
                rotated_image = cv2.warpAffine(image, M, (w, h))

                output_path = os.path.join(product_dir, f'{obj}_version_{img_name.split(".")[0]}_{j}.jpg')
                cv2.imwrite(output_path, rotated_image)

    cv2.destroyAllWindows()
