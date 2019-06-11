import os
import cv2
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from config import CarConfig

# Crop train and test according to bounding box in cars_train_annos.mat and cars_test_annos.mat,
# That making model can focus on car, reduce the effect of the background
if __name__ == "__main__":
    car = CarConfig()

    # Crop train image, cross validation
    if not os.path.exists(car.conf.train_crop_dir):
        os.makedirs(car.conf.train_crop_dir)

    mat = loadmat(car.conf.train_annos)
    anns = mat['annotations'][0]
    print('TRAIN SIZE:', len(anns))

    image_ids = []
    class_ids = []
    for ann in tqdm(anns):
        img_id = ann[5][0]
        class_id = int(ann[4][0][0])
        image_ids.append(img_id)
        class_ids.append(class_id)

        x1 = int(ann[0][0])
        y1 = int(ann[1][0])
        x2 = int(ann[2][0])
        y2 = int(ann[3][0])

        # Save crop image
        image_path = os.path.join(car.conf.train_dir, img_id)
        img = cv2.imread(image_path)
        img = img[y1:y2, x1:x2]
        new_image_path = image_path.replace(car.conf.train_dir, car.conf.train_crop_dir )
        cv2.imwrite(new_image_path, img)

    df = pd.DataFrame()
    df['img_id'] = np.array(image_ids)
    df['cls_id'] = np.array(class_ids)

    folds = [0]*df.shape[0]
    # split trainset to 5 folds, using 4 folds for training and 1 fold for evaluation
    skf = KFold(n_splits=car.conf.folds, random_state=8, shuffle=True)
    for fold, (_, valid_indexs) in enumerate(skf.split(df.index.values)):
        for vi in valid_indexs:
            folds[vi] = fold
    df['fold'] = np.array(folds)

    if not os.path.isfile(car.conf.trainset):
        df.to_csv(car.conf.trainset, index=False)

    # Crop test image
    if not os.path.exists(car.conf.test_crop_dir):
        os.makedirs(car.conf.test_crop_dir)

    mat = loadmat(car.conf.test_annos)
    anns = mat['annotations'][0]
    print('TEST SIZE:', len(anns))
    for ann in tqdm(anns):
        img_id = ann[4][0]
        x1 = int(ann[0][0])
        y1 = int(ann[1][0])
        x2 = int(ann[2][0])
        y2 = int(ann[3][0])

        # Save crop image
        image_path = os.path.join(car.conf.test_dir, img_id)
        img = cv2.imread(image_path)
        img = img[y1:y2, x1:x2]
        new_image_path = image_path.replace(car.conf.test_dir, car.conf.test_crop_dir )
        cv2.imwrite(new_image_path, img)