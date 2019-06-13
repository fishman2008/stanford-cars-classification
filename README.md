# Stanford-cars classification

In this repository, I'm making a cars classifier using the Stanford cars dataset, which contains 196 classes(including make and model). This repository also contains the checkpoint of 13 models trained on Stanford-cars dataset with high accuracy. You can use it as pretrained weights then transfer learning on others dataset.<br />
Ensemble of some models in this repository can achieve accuracy [**0.9462**](https://github.com/dungnb1333/stanford-cars-classification/blob/master/submission/Ensemble.txt), higher accuracy than [state-of-the-art stanford cars 2018](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford) (**0.945**) and nearly [state-of-the-art image classification on stanford cars 2019](https://paperswithcode.com/sota/image-classification-on-stanford-cars) (**0.947**)

## Environments
- Ubuntu 16.04 LTS
- Cuda 10.0, cuDNN v7.5.0
- Python 3.5, Keras 2.2.4, Tensorflow 1.13.1, Efficientnet
- Quick install dependencies:<br />$ **pip install --upgrade -r requirement.txt**

## Datasets

https://ai.stanford.edu/~jkrause/cars/car_dataset.html<br />
<sub>3D Object Representations for Fine-Grained Categorization<br />
Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei<br />
4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.<br />
[[pdf]](https://ai.stanford.edu/~jkrause/papers/3drr13.pdf) [[BibTex]](https://ai.stanford.edu/~jkrause/papers/3drr13.bib) [[slides]](https://ai.stanford.edu/~jkrause/papers/3drr_talk.pdf)

196 classes<br />
Trainset: 8144 images<br />
Testset: 8041 images<br />
Some images in training set:
<p align="center">
  <img src="https://github.com/dungnb1333/stanford-cars-classification/raw/master/images/train_samples.png">
</p>
Distribution of training set:
<p align="center">
  <img src="https://github.com/dungnb1333/stanford-cars-classification/raw/master/images/distribution.png">
</p>
Min: 24 images/class , max: 68 images/class , mean: 41 images/class, so this dataset is quite balanced.<br /><br />

Quick download datasets via command line:<br />
$ **bash quick_download.sh**<br />
Cross-validation 5 folds<br />
$ **python prepare.py**<br />

## Training

Using pre-trained weights on imagenet dataset, with transfer learning to train the model. All layers will be fine tuned and the last fully connected layer will be replaced entirely.
Useful tricks I used for training:
- Cyclical Learning Rate [[paper]](https://arxiv.org/abs/1506.01186) [[repo]](https://github.com/bckenstler/CLR)
- Heavy augmentation: random crops, horizontal flip, rotate, shear, AddToHueAndSaturation, AddMultiply, GaussianBlur, ContrastNormalization, sharpen, emboss
- Random eraser [[paper]](https://arxiv.org/abs/1708.04896)
- Mixup [[paper]](https://arxiv.org/abs/1710.09412)
- Cross-validation 5 folds

$ **python train.py --network network --gpu gpu_id --epochs number_of_epochs --multiprocessing False/True**<br />
You can choose any network in list:<br />
- VGG16, VGG19
- ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, ResNeXt50, ResNeXt101
- InceptionV3, InceptionResNetV2, Xception
- MobileNet, MobileNetV2
- DenseNet121, DenseNet169, DenseNet201
- NASNetMobile, NASNetLarge
- EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

For example to train MobileNetV2 on 200 epochs:<br />
$ **python train.py --network MobileNetV2 --gpu 0 --epochs 200 --multiprocessing False**<br />

I used the optimal parameters (input size, batch_size) for my hardware (1x1080 Ti 12GB, RAM 32GB, CPU 12 Core), you can modify [config.py](https://github.com/dungnb1333/stanford-cars-classification/blob/master/config.py) to suit your hardware.

I saved training log of 13 models on each fold in [logs](https://github.com/dungnb1333/stanford-cars-classification/tree/master/logs)

## Checkpoint

Download checkpoint of 13 models in [link](https://www.dropbox.com/sh/jv7dbd5ksj2exun/AAATZFgaxe7rMEjv10PG1BYha?dl=0) then put into [checkpoints](https://github.com/dungnb1333/stanford-cars-classification/tree/master/checkpoints) to evaluate model, generate submission or demo on image.

## Evaluate models:

To enhance the result, I applied 12 crops for validation and test prediction, accuracy of single model is ensemble of 12 crops and 5 folds. For example with input shape of network is 224x224x3:<br />
<p align="center">
  <img src="https://github.com/dungnb1333/stanford-cars-classification/raw/master/images/12crops.png">
</p>

To evaluate network, run:<br />
$ **python evaluate.py --network network --gpu gpu_id --multi_crops True/False**<br />
For example:<br />
$ **python evaluate.py --network MobileNetV2 --gpu 0 --multi_crops True**<br />

To generate submission for each model, run:<br />
$ **python predict.py --network network --gpu gpu_id**

Output is network.txt in folder [submission](https://github.com/dungnb1333/stanford-cars-classification/tree/master/submission) and raw output network.npy in folder [data](https://github.com/dungnb1333/stanford-cars-classification/tree/master/data)

You can summit your result at [stanford-cars evaluation server](http://imagenet.stanford.edu/internal/car196/submission/submission.php).

Accuracy and size of 13 models:<br />
<p align="center">
  <img src="https://github.com/dungnb1333/stanford-cars-classification/raw/master/images/model_accuracy.png">
</p>

## Ensemble multi-models

Final result [**0.9462**](https://github.com/dungnb1333/stanford-cars-classification/blob/master/submission/Ensemble.txt) is ensemble of some models with suitable ratios: result = sum(weight x model) / sum(weight).<br />
$ **python ensemble.py**<br />

I just tried a few cases, you can try with other ratios and other models to get higher accuracy than 0.9462.

## Demo on image

$ **python demo.py --network network --gpu gpu_id --image_path path --imshow True/False**<br />
For example:<br />
$ **python demo.py --network ResNeXt101 --gpu 0 --image_path images/samples/02381.jpg --imshow True**<br />
<p align="center">
  <img src="https://github.com/dungnb1333/stanford-cars-classification/raw/master/images/demo.png">
</p>