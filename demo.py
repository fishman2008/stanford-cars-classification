import argparse
from models import Car_Model, batch_preprocess_input
from config import CarConfig
import pandas as pd
import numpy as np
import cv2
import os
from keras.utils import Sequence
import keras
from keras.models import *
from scipy.io import loadmat
import matplotlib.pyplot as plt

def transform(conf, img_org, type):
    if type == 0 or type == 1:
        img = cv2.resize(img_org, (conf.size, conf.size), interpolation = cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img_org, (conf.tmp_size, conf.tmp_size), interpolation = cv2.INTER_CUBIC)
    if type%2 == 1:
        img = np.flip(img, axis=1)
    if type == 2 or type == 7:
        img = img[0:conf.size, 0:conf.size]
    elif type == 3 or type == 8:
        img = img[0:conf.size, conf.tmp_size - conf.size:conf.tmp_size]
    elif type == 4 or type == 9:
        img = img[conf.tmp_size - conf.size:conf.tmp_size, 0:conf.size]
    elif type == 5 or type == 10:
        img = img[conf.tmp_size - conf.size:conf.tmp_size, conf.tmp_size - conf.size:conf.tmp_size]
    elif type == 6 or type == 11:
        p = int((conf.tmp_size - conf.size)/2)
        img = img[p:p+conf.size,p:p+conf.size]
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for testing')
    parser.add_argument("--image_path", help="path of image", default="images/samples/02381.jpg", type=str)
    parser.add_argument("--gpu", help="GPU id", default="0", type=str)
    parser.add_argument("--network", help="which network", default="MobileNetV2", type=str)
    parser.add_argument("--imshow", default=False, type=lambda x: (str(x).lower() == "true"))
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    car = CarConfig()
    car.update(args.network)

    mat = loadmat(car.conf.cars_meta)
    class_names = mat['class_names'][0]

    model = Car_Model(base_model_name = car.conf.network, size = car.conf.size, pool = car.conf.pool, class_nums = car.conf.class_nums)

    img_org = cv2.imread(args.image_path)
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    imgs = []
    for type in range(12):
        imgs.append(transform(car.conf, img_org, type))
    imgs = np.array(imgs, dtype = np.float32)
    imgs = batch_preprocess_input(imgs, car.conf.network)

    predict = np.zeros(car.conf.class_nums, dtype = np.float64)
    for fold in range(car.conf.folds):
        BEST_WEIGHTS = 'checkpoints/%s_weight_fold%d.hdf5'%(car.conf.network, fold)
        model.load_weights(BEST_WEIGHTS)
        predict += np.mean(model.predict(imgs),axis=0)
    predict /= float(car.conf.folds)

    idx = np.argmax(predict)
    class_id = idx + 1
    prob = predict[idx]
    name = class_names[idx][0]

    print('ClassID: %d\nName: %s\Confidence: %f'%(class_id, name, prob))
    
    if args.imshow:
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB))
        fig.suptitle('%s    Confidence: %f'%(name, prob))
        plt.show()