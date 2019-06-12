from easydict import EasyDict as edict

class CarConfig():
    def __init__(self):
        self.conf = edict()
        self.conf.class_nums = 196
        self.conf.folds = 5
        self.conf.train_dir = 'datasets/cars_train'
        self.conf.test_dir = 'datasets/cars_test'
        self.conf.train_annos = 'datasets/devkit/cars_train_annos.mat'
        self.conf.cars_meta = 'datasets/devkit/cars_meta.mat'
        self.conf.test_annos = 'datasets/devkit/cars_test_annos.mat'

        self.conf.train_crop_dir = 'datasets/crop_train'
        self.conf.test_crop_dir = 'datasets/crop_test'
        self.conf.trainset = 'datasets/trainset.csv'

    def update(self, network = 'MobileNetV2'):
        self.conf.network = network

        if  self.conf.network == 'Xception':
            self.conf.size = 320
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'Concat'
            self.conf.batch_size = 16

        if  self.conf.network == 'VGG16'       or \
            self.conf.network == 'VGG19'       or \
            self.conf.network == 'ResNet50'    or \
            self.conf.network == 'ResNet101'   or \
            self.conf.network == 'ResNet152'   or \
            self.conf.network == 'ResNet50V2'  or \
            self.conf.network == 'ResNet101V2' or \
            self.conf.network == 'ResNet152V2' or \
            self.conf.network == 'ResNeXt50'   or \
            self.conf.network == 'ResNeXt101':
            self.conf.size = 224
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 16
        
        if  self.conf.network == 'DenseNet121' or \
            self.conf.network == 'DenseNet169' or \
            self.conf.network == 'DenseNet201':
            self.conf.size = 224
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 20

        if  self.conf.network == 'MobileNet'   or \
            self.conf.network == 'MobileNetV2':
            self.conf.size = 384
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 32

        if  self.conf.network == 'InceptionV3':
            self.conf.size = 416
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalMaxPooling2D'
            self.conf.batch_size = 32

        if  self.conf.network == 'InceptionResNetV2':
            self.conf.size = 299
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalMaxPooling2D'
            self.conf.batch_size = 16

        if  self.conf.network == 'NASNetLarge':
            self.conf.size = 331
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 4

        if  self.conf.network == 'EfficientNetB0':
            self.conf.size = 224
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 64

        if  self.conf.network == 'EfficientNetB1':
            self.conf.size = 240
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 32

        if  self.conf.network == 'EfficientNetB2':
            self.conf.size = 260
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 32

        if  self.conf.network == 'EfficientNetB3':
            self.conf.size = 300
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 16
        
        if  self.conf.network == 'EfficientNetB4':
            self.conf.size = 380
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 1

        if  self.conf.network == 'EfficientNetB5':
            self.conf.size = 456
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 1

        if  self.conf.network == 'EfficientNetB6':
            self.conf.size = 528
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 1

        if  self.conf.network == 'EfficientNetB7':
            self.conf.size = 600
            self.conf.tmp_size = int(1.25*self.conf.size)
            self.conf.pool = 'GlobalAveragePooling2D'
            self.conf.batch_size = 1
