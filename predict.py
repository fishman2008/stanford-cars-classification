import argparse
from models import Car_Model
from config import CarConfig
import pandas as pd
import numpy as np
import cv2
import os
from random import randint
from tqdm import tqdm
import efficientnet
from keras.utils import Sequence
from keras_applications import xception, vgg16, vgg19, resnet, resnet_v2, resnext, inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2, densenet, nasnet
import keras
from keras.models import *

TEST_NUMS = 8041
def batch_preprocess_input(x_batch, network):
    if network == 'Xception':
        x_batch = xception.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'VGG16':
        x_batch = vgg16.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'VGG19':
        x_batch = vgg19.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'ResNet50' or network == 'ResNet101' or network == 'ResNet152':
        x_batch = resnet.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'ResNet50V2' or network == 'ResNet101V2' or network == 'ResNet152V2':
        x_batch = resnet_v2.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'ResNeXt50' or network == 'ResNeXt101':
        x_batch = resnext.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'InceptionV3':
        x_batch = inception_v3.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'InceptionResNetV2':
        x_batch = inception_resnet_v2.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'MobileNet':
        x_batch = mobilenet.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'MobileNetV2':
        x_batch = mobilenet_v2.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'DenseNet121' or network == 'DenseNet169' or network == 'DenseNet201':
        x_batch = densenet.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif network == 'NASNetMobile' or network == 'NASNetLarge':
        x_batch = nasnet.preprocess_input(x_batch, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif 'EfficientNet' in network:
        x_batch = efficientnet.preprocess_input(x_batch)
    else:
        return None

    return x_batch

class PredictGenerator(Sequence):
    def __init__(self, conf, number_of_images, dict, batch_size, augtype):
        self.conf = conf
        self.number_of_images = number_of_images
        self.dict = dict
        self.batch_size = batch_size
        self.augtype = augtype

    def __len__(self):
        return int(np.ceil(float(self.number_of_images) / self.batch_size))

    def __getitem__(self, index):
        start = index*self.batch_size + 1
        end = min((index+1)*self.batch_size, self.number_of_images) + 1
        x_batch = []
        for i in range(start,end,1):
            img = self.dict[i]
            if self.augtype == 0 or self.augtype == 1:
                img = cv2.resize(img, (self.conf.size, self.conf.size), interpolation = cv2.INTER_CUBIC)
            else:
                img = cv2.resize(img, (self.conf.tmp_size, self.conf.tmp_size), interpolation = cv2.INTER_CUBIC)
            if self.augtype%2 == 1:
                img = np.flip(img, axis=1)
            if self.augtype == 2 or self.augtype == 7:
                img = img[0:self.conf.size, 0:self.conf.size]
            elif self.augtype == 3 or self.augtype == 8:
                img = img[0:self.conf.size, self.conf.tmp_size - self.conf.size:self.conf.tmp_size]
            elif self.augtype == 4 or self.augtype == 9:
                img = img[self.conf.tmp_size - self.conf.size:self.conf.tmp_size, 0:self.conf.size]
            elif self.augtype == 5 or self.augtype == 10:
                img = img[self.conf.tmp_size - self.conf.size:self.conf.tmp_size, self.conf.tmp_size - self.conf.size:self.conf.tmp_size]
            elif self.augtype == 6 or self.augtype == 11:
                p = int((self.conf.tmp_size - self.conf.size)/2)
                img = img[p:p+self.conf.size,p:p+self.conf.size]

            x_batch.append(img)

        x_batch = np.array(x_batch, dtype = np.float32)
        x_batch = batch_preprocess_input(x_batch, self.conf.network)

        return x_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for testing')
    parser.add_argument("--gpu", help="GPU id", default="0", type=str)
    parser.add_argument("--network", help="which network", default="MobileNetV2", type=str)
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    car = CarConfig()
    car.update(args.network)

    model = Car_Model(base_model_name = car.conf.network, size = car.conf.size, pool = car.conf.pool, class_nums = car.conf.class_nums)

    if not os.path.exists('submission'):
        os.makedirs('submission')

    print('Load images to dictionary ...')
    test_dict = {}
    for test_id in tqdm(range(1,TEST_NUMS+1,1)):
        img_path = os.path.join(car.conf.test_crop_dir, '%05d.jpg'%test_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_dict[test_id] = img

    ptest_ensemble = np.zeros((TEST_NUMS, car.conf.class_nums), dtype=np.float64)
    print('Ensembling 5 fols and 12 crops...')
    for fold in range(car.conf.folds):
        print('Fold %d'%fold)
        BEST_WEIGHTS = 'checkpoints/%s_weight_fold%d.hdf5'%(car.conf.network, fold)
        model.load_weights(BEST_WEIGHTS)
        for type in range(12):
            ptest_ensemble += model.predict_generator(generator = PredictGenerator(conf=car.conf, number_of_images = TEST_NUMS, dict=test_dict, batch_size=64, augtype=type), verbose=1)
    ptest_ensemble /= float(12*car.conf.folds)
    
    ypredict = np.argmax(ptest_ensemble, axis=1) + 1
    submission_file = open('submission/%s.txt'%args.network, 'w')
    submission_file.write('\n'.join(map(str, ypredict.tolist())))
    submission_file.close()

    del model
    keras.backend.clear_session()