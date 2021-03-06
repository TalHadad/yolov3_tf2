# raspberry_pi_video.py

import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolov3 import yolov3_net
import cv2
import time
from datetime import datetime
import picamera
import PIL
model_size = (416, 416, 3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 100
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfg_file = 'cfg/yolov3.cfg'
weight_file = 'weights/yolov3_weights.tf'

def analyze_cv_single_picture():
    model = yolov3_net(cfg_file, model_size, num_classes)
    model.load_weights(weight_file)

    class_names = load_class_names(class_name)
    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)
    cap = cv2.VideoCapture(0)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    try:
        while True:
            start = time.time()

            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            cap.release()

            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)
            boxes, scores, classes, nums = output_boxes(pred,
                                                        model_size,
                                                        max_output_size = max_output_size,
                                                        max_output_size_per_class = max_output_size_per_class,
                                                        iou_threshold = iou_threshold,
                                                        confidence_threshold = confidence_threshold)
            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)

            stop = time.time()
            seconds = stop - start
            print(f'Time taken : {seconds} seconds')

            # Calcutate frames per seconds
            fps = 1 / seconds
            print(f'Estimated frames per second : {fps}')

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        print('Detections have been performed successfully.')

# still slow, 20 sec for image
def analyze_cv_live_stream_single_frames():
    model = yolov3_net(cfg_file, model_size, num_classes)
    model.load_weights(weight_file)

    class_names = load_class_names(class_name)
    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)


    try:
        while True:
            cap = cv2.VideoCapture(0)
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)
            boxes, scores, classes, nums = output_boxes(pred,
                                                        model_size,
                                                        max_output_size = max_output_size,
                                                        max_output_size_per_class = max_output_size_per_class,
                                                        iou_threshold = iou_threshold,
                                                        confidence_threshold = confidence_threshold)
            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)

            stop = time.time()
            seconds = stop - start
            print(f'Time taken : {seconds} seconds')

            # Calcutate frames per seconds
            fps = 1 / seconds
            print(f'Estimated frames per second : {fps}')

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            cap.release()
    finally:
        cv2.destroyAllWindows()
        print('Detections have been performed successfully.')

# very slow, 30 sec for frame
def analyze_picamera_single_frames():
    model = yolov3_net(cfg_file, model_size, num_classes)
    model.load_weights(weight_file)

    class_names = load_class_names(class_name)
    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    # capture() for images
    # start_recording() for video
    # start_preview() for live stream
    try:
        with picamera.PiCamera() as camera:
            while True:
                start = time.time()
                now = datetime.now()
                camera.annotate_text = str(now)
                camera.capture('data/images/raspi.jpeg', format='jpeg')
                #frame = PIL.Image.open('data/images/raspi.jpeg')
                frame = cv2.imread('data/images/raspi.jpeg')
                cv2.imshow(win_name, frame)
                # analyzed_image = analyze_image(image)

                resized_frame = tf.expand_dims(frame, 0)
                resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

                pred = model.predict(resized_frame)
                boxes, scores, classes, nums = output_boxes(pred,
                                                            model_size,
                                                            max_output_size = max_output_size,
                                                            max_output_size_per_class = max_output_size_per_class,
                                                            iou_threshold = iou_threshold,
                                                            confidence_threshold = confidence_threshold)
                img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
                cv2.imshow(win_name, img)

                stop = time.time()
                seconds = stop - start
                print(f'Time taken : {seconds} seconds')

                # Calcutate frames per seconds
                fps = 1 / seconds
                print(f'Estimated frames per second : {fps}')

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

    finally:
        cv2.destroyAllWindows()
        print('Detections have been performed successfully.')

# slow
def analyze_cv_live_stream():
    model = yolov3_net(cfg_file, model_size, num_classes)
    model.load_weights(weight_file)

    class_names = load_class_names(class_name)
    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    # specify the vidoe input.
    # 0 means input from cam 0.
    # For video, change the 0 to video path
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('data/videos/cats.mp4')
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)
            boxes, scores, classes, nums = output_boxes(pred,
                                                        model_size,
                                                        max_output_size = max_output_size,
                                                        max_output_size_per_class = max_output_size_per_class,
                                                        iou_threshold = iou_threshold,
                                                        confidence_threshold = confidence_threshold)
            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)

            stop = time.time()
            seconds = stop - start
            print(f'Time taken : {seconds} seconds')

            # Calcutate frames per seconds
            fps = 1 / seconds
            print(f'Estimated frames per second : {fps}')

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')

if __name__ == '__main__':
    analyze_cv_single_picture()
