# Semantic Segmentation

### Update
* MobileNetV1 -> MobileNetV2
* Docker image
* Updated requirements.txt

### Overview
This project is part of the Udacity Car-ND. Originally, it uses a VGG-frontend. However, VGG is old, slow and uses too much memory. E.g. a single (160, 576) image already requires 4GB GPU memory. I therefore switched to the [MobileNet](https://arxiv.org/abs/1704.04861) architecture. All stride-16 depthwise convolutions were replaced with dilated depthwise convolutions and the two final stride-32 layers were removed. This is similar to what was done in [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122).
Though, I added skip-connections for stride-8 and stride-4 (adding stride-2 gave no better results). 
The model uses the [Keras MobileNet implementation](https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py) and training is done with
[TensorFlow](https://www.tensorflow.org/).
Since the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) provides no official training/validation splits I used 10% of the training data for validation. Images are downscaled to half resolution.

[//]: # (Image References)
[image1]: ./res/loss_curves.png
[image2]: ./res/augmentation_methods_overview.png
[image3]: ./res/latest_run.png
[image5]: ./res/highway.gif

#### Data augmentation
The dataset is rather small i.e. only 289 training images. Thus, I used a few augmentation methods to generate more data. These methods include: rotation, flipping, blurring and changing the illumination of the scene (see `augmentation.py`).
An example is given in the following image:
![alt text][image2]

#### Quantitative Results
Training for 40 epochs results in the following loss curves:
![alt text][image1]
I ran a few experiments with different learning rate schedules, varying amounts of data augmentation and changed dilation rates but the results did not change that much. Moreover, there was no big difference in training from scratch vs. using ImageNet weights.
By default logs are saved to the `log_dir` directory. To visualize them start tensorboard via `tensorboard --logdir log_dir`.

#### Qualitative Results
Here are a few predictions @~90% validation IoU:
![alt text][image3]
It can be seen that shadows are handled quite well. Road vs. sidewalk still leaves room for improvements (e.g. bottom left). More results can be found in the `val_predictions` directory.

### Setup

#### Clone this repository
```console
$ git clone git@github.com:see--/P12-Semantic-Segmentation.git && cd P12-Semantic-Segmentation
```

#### Get the data
```console
# http://www.cvlibs.net/download.php?file=data_road.zip
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip
$ unzip data_road && rm data_road.zip
```

#### Build the docker image
```console
# without gpu
$ docker build -t roadeye:latest -f Dockerfile .
# with gpu: https://www.tensorflow.org/install/docker#download_a_tensorflow_docker_image
$ docker build -t roadeye:latest -f Dockerfile-gpu .
```

#### Start the docker image
```console
# without gpu
$ docker run -v $PWD:/road_segmentation -w /road_segmentation -it --rm roadeye:latest
# with gpu
$ docker run --runtime=nvidia -v $PWD:/road_segmentation -w /road_segmentation -it --rm roadeye:latest
```

#### Train
```console
$ python src/train.py
```
You can skip the Train step and download the pre-trained model from the 'release' tab.
https://github.com/see--/P12-Semantic-Segmentation/releases/download/v0.1.0/best-ep-38-val_iou-0.898-val_loss-0.064.hdf5

#### Convert the model to tflite format
Use the hdf5 with the highest `val_iou`.

```console
$ tflite_convert  \
  --output_file=roadeye.tflite \
  --keras_model_file best-ep-38-val_iou-0.898-val_loss-0.064.hdf5
```

#### Predict
```
# validation dataset
python src/dump_predictions.py --model best-ep-38-val_iou-0.898-val_loss-0.064.hdf5
# custom dataset
python src/dump_predictions.py --model best-ep-38-val_iou-0.898-val_loss-0.064.hdf5 --fn-glob "video_jpgs/*.jpg" --crop 150,-100
```


### Generalization
We only trained on images from Germany. Here are a few results for Melbourne Australia.
<p align="center">
    <img src="res/highway.gif">
</p>
