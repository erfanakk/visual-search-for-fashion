
# object detection top clothing
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-360/)

- Object Detection using [TensorFlow-Object-Detection_API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

- Object detection allows for the recognition, detection of multiple objects within an image. It provides us a much better understanding of an image as a whole as opposed to just visual recognition.


Model name                                                                                                                                                                  | Speed (ms) | COCO mAP | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :----------: | :-----:
[EfficientDet D0 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz)                                  | 39         | 33.6           | Boxes
[EfficientDet D1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)                                  | 54         | 38.4           | Boxes
[EfficientDet D2 768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz)                                  | 67         | 41.8           | Boxes
[EfficientDet D3 896x896](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz)                                  | 95         | 45.4           | Boxes
[EfficientDet D4 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz)                              | 133         | 48.5           | Boxes
[EfficientDet D5 1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz)                             | 222         | 49.7           | Boxes
[EfficientDet D6 1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz)                             | 268         | 50.5           | Boxes
[EfficientDet D7 1536x1536](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz)                             | 325         | 51.2           | Boxes


## Overview
`Deep Clothes Detector` is a clothes detection framework based on [EfficientDet D3([https://github.com/rbgirshick/fast-rcnn](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz). Given a fashion image, this software finds and localizes potential *upper-body clothes*




<img src="images/img1.jpg" width="300" height="300"/> <img src="images/img2.jpg" width="300" height="300"/> <img src="images/img3.jpg" width="300" height="300"/> 


## Requirements

* **python 3.8**
* **opencv (cv2)**
* **tensorboard**
* **pycocotools**
* **efficientnet_d3**

