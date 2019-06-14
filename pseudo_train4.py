import argparse
from models import Car_Model, batch_preprocess_input
from config import CarConfig
import pandas as pd
import numpy as np
import cv2
import os
from random import randint
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
from clr_callback import CyclicLR
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from keras.utils import Sequence
import keras
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.models import *

def random_resize_crop(img, tmp_size, dst_size):
    img = cv2.resize(img, (tmp_size, tmp_size), interpolation = cv2.INTER_CUBIC)
    x = randint(0, tmp_size - dst_size)
    y = randint(0, tmp_size - dst_size)
    crop_img = img[y:y+dst_size, x:x+dst_size]
    return crop_img

def random_eraser(img, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    img_h, img_w, _ = img.shape
    p_1 = np.random.rand()
    if p_1 > p:
        return img
    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        if left + w <= img_w and top + h <= img_h:
            break
    c = np.random.uniform(v_l, v_h)
    img[top:top + h, left:left + w, :] = c
    return img

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    sometimes(
        iaa.OneOf([
            iaa.Affine(
                rotate=(-15, 15),
                shear=(-10, 10),
                cval=0
            ),
            iaa.AddToHueAndSaturation((-20, 20)),
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.GaussianBlur((0, 1.5)),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
            iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))
        ])
    )
])

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    sample_size = x.shape[0]
    index_array = np.arange(sample_size)
    np.random.shuffle(index_array)
    mixed_x = lam * x + (1 - lam) * x[index_array]
    mixed_y = (lam * y) + ((1 - lam) * y[index_array])
    return mixed_x, mixed_y

class TrainGenerator(Sequence):
    def __init__(self, conf, train_df, train_dict, external_dict, batch_size):
        self.conf = conf
        self.train_df = train_df
        self.train_dict = train_dict
        self.external_dict = external_dict
        self.batch_size = batch_size
        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return int(np.ceil(float(self.train_df.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        start = index*self.batch_size
        end = min((index+1)*self.batch_size, self.train_df.shape[0])
        sub_df = self.train_df.iloc[start:end]
        x_batch = []
        y_batch = []
        for _, row in sub_df.iterrows():
            if row['isExternal'] == 1:
                img = self.external_dict[row['img_id']]
            else:
                img = self.train_dict[row['img_id']]
            img = random_resize_crop(img, self.conf.tmp_size, self.conf.size)
            img = seq.augment_image(img)
            img = random_eraser(img, p=0.5)
            x_batch.append(img)
            y_batch.append(to_categorical(row['cls_id']-1, num_classes=self.conf.class_nums))
        x_batch = np.array(x_batch, dtype = np.float32)
        x_batch = batch_preprocess_input(x_batch, self.conf.network)
        y_batch = np.array(y_batch, np.int)
        x_batch,y_batch = mixup_data(x_batch,y_batch,alpha=0.5)
        return x_batch, y_batch

    def on_epoch_end(self):
        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)

class ValidGenerator(Sequence):
    def __init__(self, conf, valid_df, valid_dict, batch_size):
        self.conf = conf
        self.valid_df = valid_df
        self.valid_dict = valid_dict
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(float(self.valid_df.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        start = index*self.batch_size
        end = min((index+1)*self.batch_size, self.valid_df.shape[0])
        sub_df = self.valid_df.iloc[start:end]
        x_batch = []
        y_batch = []
        for _, row in sub_df.iterrows():
            img = self.valid_dict[row['img_id']]
            x_batch.append(img)
            y_batch.append(to_categorical(row['cls_id']-1, num_classes=self.conf.class_nums))
        x_batch = np.array(x_batch, dtype = np.float32)
        x_batch = batch_preprocess_input(x_batch, self.conf.network)
        y_batch = np.array(y_batch, np.int)
        return x_batch, y_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for training')
    parser.add_argument("--epochs", help="training epochs", default=200, type=int)
    parser.add_argument("--multiprocessing", help="multiprocessing", default=False, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--gpu", help="GPU id", default="0", type=str)
    parser.add_argument("--network", help="which network", default='MobileNetV2', type=str)
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    car = CarConfig()
    car.update(args.network)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    df = pd.read_csv(car.conf.trainset)
    external_df = pd.read_csv('datasets/external_dataset.csv')
    external_df = external_df.loc[:, ['img_id', 'cls_id']]
    external_df['isExternal'] = np.ones(external_df.shape[0], dtype=np.uint8)

    for fold in range(4,5,1):
        valid_df = df.loc[df['fold'] == fold]
        train_df = df.loc[~df.index.isin(valid_df.index)]
        train_df = train_df.loc[:, ['img_id', 'cls_id']]
        train_df['isExternal'] = np.zeros(train_df.shape[0], dtype=np.uint8)
        train_df = pd.concat([train_df, external_df], ignore_index=True)

        train_size = train_df.shape[0]
        valid_size = valid_df.shape[0]

        print('Load images to dictionary ...')
        train_dict = {}
        external_dict = {}
        for _, row in tqdm(train_df.iterrows(), total = train_size):
            if row['isExternal'] == 1:
                img_path = row['img_id']
            else:
                img_path = os.path.join(car.conf.train_crop_dir, row['img_id'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if row['isExternal'] == 1:
                external_dict[row['img_id']] = img
            else:
                train_dict[row['img_id']] = img

        valid_dict = {}
        for _, row in tqdm(valid_df.iterrows(), total = valid_size):
            img_path = os.path.join(car.conf.train_crop_dir, row['img_id'])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (car.conf.size, car.conf.size), interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            valid_dict[row['img_id']] = img

        train_steps = np.ceil(float(train_size) / float(car.conf.batch_size))

        BEST_WEIGHTS = 'checkpoints/%s_pseudo_weight_fold%d.hdf5'%(car.conf.network, fold)
        TRAINING_LOG = 'logs/%s_pseudo__trainlog_fold%d.csv'%(car.conf.network, fold)

        clr = CyclicLR(base_lr=1e-7, max_lr=2e-4, step_size=4*train_steps, mode='exp_range',gamma=0.99994)
        early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
        save_checkpoint = ModelCheckpoint(BEST_WEIGHTS, monitor = 'val_acc', verbose = 1, save_weights_only = True, save_best_only=True, mode='max')
        csv_logger = CSVLogger(TRAINING_LOG, append=False)
        callbacks = [early_stopping, save_checkpoint, csv_logger, clr]

        print('FOLD %d    TRAIN SIZE: %d    VALID SIZE: %d'%(fold, train_size, valid_size))

        model = Car_Model(base_model_name = car.conf.network, size = car.conf.size, pool = car.conf.pool, class_nums = car.conf.class_nums)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=2e-4), metrics=['accuracy'])
        model.summary()
        model.fit_generator(generator=TrainGenerator(conf = car.conf, train_df = train_df, train_dict = train_dict, external_dict = external_dict, batch_size = car.conf.batch_size),
                            validation_data=ValidGenerator(conf = car.conf, valid_df = valid_df, valid_dict =valid_dict, batch_size=car.conf.batch_size),
                            callbacks = callbacks,
                            epochs=args.epochs, 
                            verbose=1,
                            max_queue_size=8, 
                            workers=4, 
                            use_multiprocessing=args.multiprocessing)
        del model
        keras.backend.clear_session()