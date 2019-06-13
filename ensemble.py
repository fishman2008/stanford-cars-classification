from config import CarConfig
import pandas as pd
import os
import numpy as np

models_and_weights = {
    'EfficientNetB1':   0.021194,
    'EfficientNetB2':   0.087494,
    'EfficientNetB3':   0.109124,
    'ResNeXt101':       0.189314,
    'DenseNet201':      0.273687,
    'ResNet152V2':      0.013186,
    'NASNetLarge':      0.306000,
}

TEST_NUMS = 8041

if __name__ == '__main__':
    car = CarConfig()

    xtest_ensemble = np.zeros((TEST_NUMS, car.conf.class_nums), dtype=np.float64)
    for mset, weight in models_and_weights.items():
        xtest_ensemble += weight*np.load('data/%s.npy'%mset)
    xtest_ensemble /= float(car.conf.folds)

    ypredict = np.argmax(xtest_ensemble, axis=1) + 1
    submission_file = open('submission/Ensemble.txt', 'w')
    submission_file.write('\n'.join(map(str, ypredict.tolist())))
    submission_file.close()