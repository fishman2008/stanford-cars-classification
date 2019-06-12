from config import CarConfig
import pandas as pd
import os
import numpy as np

models_and_weights = {
    'EfficientNetB0':       0.0037,
    'EfficientNetB1':       0.0212,
    'EfficientNetB2':       0.0938,
    'EfficientNetB3':       0.1183,
    'ResNeXt101':           0.2932,
    'DenseNet201':          0.0988,
    'InceptionResNetV2':    0.0128,
    'Xception':             0.0001,
    'ResNet152V2':          0.0558,
    'MobileNetV2':          0.0253,
    'NASNetLarge':          0.2770,
}

TEST_NUMS = 8041

if __name__ == '__main__':
    car = CarConfig()

    xtest_ensemble = np.zeros((TEST_NUMS, car.conf.class_nums), dtype=np.float64)
    for mset, weight in models_and_weights.items():
        xtest_ensemble += weight*np.load('data/%s.npy'%mset)
    xtest_ensemble /= float(car.conf.folds)

    ypredict = np.argmax(xtest_ensemble, axis=1) + 1
    submission_file = open('submission/blend_submission.txt', 'w')
    submission_file.write('\n'.join(map(str, ypredict.tolist())))
    submission_file.close()