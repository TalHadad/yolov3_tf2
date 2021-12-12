# yolov3_tf2
Implementation of Yolo v3 with tensorflow 2.*

Download weights from:
https://pjreddie.com/media/files/yolov3.weights
and put the file in weights directory.

Create virtual environment:
$ python -m env yolov3_tf2
$ source yolov3_tf2/bin/activate

Install packages:
(yolov3_tf2)$ pip install tensorflow
(yolov3_tf2)$ pip install opencv-python
(In raspberry pi, installation is different)

(yolov3_tf2)$ pip3 install --upgrade numpy
(yolov3_tf2)$ pip3 install pillow
(yolov3_tf2)$ pip3 install "ptvsd>=4.2" (to degug in spacemacs)

Convert weights:
(yolov3_tf2)$ python convert_weights.py

Test Image:
(yolov3_tf2)$ python image.py

Test video/camera:
(yolov3_tf2)$ python video.py
