from keras_applications.xception import Xception
from keras_applications.vgg16 import VGG16
from keras_applications.vgg19 import VGG19
from keras_applications.resnet import ResNet50, ResNet101, ResNet152
from keras_applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras_applications.resnext import ResNeXt50, ResNeXt101
from keras_applications.inception_v3 import InceptionV3
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras_applications.mobilenet import MobileNet
from keras_applications.mobilenet_v2 import MobileNetV2
from keras_applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras_applications.nasnet import NASNetMobile, NASNetLarge
from efficientnet import EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras_applications import xception, vgg16, vgg19, resnet, resnet_v2, resnext, inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2, densenet, nasnet
import efficientnet

from keras_applications import *
from keras.layers import *
from keras.models import Model
import keras

def Car_Model(base_model_name = 'MobileNetV2', size = 224, pool = 'GlobalMaxPooling2D', class_nums = 196, activation_name='softmax'):
    input_tensor = Input(shape=(size, size, 3))
    if base_model_name == 'Xception':
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'VGG16':
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'VGG19':
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNet101':
        base_model = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNet152':
        base_model = ResNet152(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNet50V2':
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNet101V2':
        base_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNet152V2':
        base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNeXt50':
        base_model = ResNeXt50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'ResNeXt101':
        base_model = ResNeXt101(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'InceptionResNetV2':
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'MobileNet':
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'DenseNet169':
        base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'DenseNet201':
        base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'NASNetMobile':
        base_model = NASNetMobile(include_top=False, weights='imagenet', input_shape=(224, 224, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'NASNetLarge':
        base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(331, 331, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name == 'EfficientNetB1':
        base_model = EfficientNetB1(include_top=False, weights='imagenet', input_shape=(240, 240, 3))
    elif base_model_name == 'EfficientNetB2':
        base_model = EfficientNetB2(include_top=False, weights='imagenet', input_shape=(260, 260, 3))
    elif base_model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
    elif base_model_name == 'EfficientNetB4':
        base_model = EfficientNetB4(include_top=False, weights=None, input_shape=(380, 380, 3))
    elif base_model_name == 'EfficientNetB5':
        base_model = EfficientNetB5(include_top=False, weights=None, input_shape=(456, 456, 3))
    elif base_model_name == 'EfficientNetB6':
        base_model = EfficientNetB6(include_top=False, weights=None, input_shape=(528, 528, 3))
    elif base_model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(include_top=False, weights=None, input_shape=(600, 600, 3))
    else:
        return None

    if pool == 'GlobalMaxPooling2D':
        x = base_model(input_tensor)
        x = GlobalMaxPooling2D()(x)
    elif pool == 'GlobalAveragePooling2D':
        x = base_model(input_tensor)
        x = GlobalAveragePooling2D()(x)
    elif pool == 'Concat':
        x0 = base_model(input_tensor)
        x1 = GlobalAveragePooling2D()(x0)
        x2 = GlobalMaxPooling2D()(x0)
        x = Concatenate()([x1,x2])
    else:
        return None

    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5,name='dropout')(x)

    output_tensor = Dense(class_nums, activation=activation_name)(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

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
