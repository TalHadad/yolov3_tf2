# yolov3.py
from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, \
    Input, \
    ZeroPadding2D, \
    LeakyReLU, \
    UpSampling2D
# print(tf.__version__)
# 2.0

def parse_cfg_file(cfg_file: str):
    '''Read configuration file and parse it into list of blocks'''
    lines = read_uncommented_lines(cfg_file)
    blocks = parse_cfg_list(lines)
    return blocks

def read_uncommented_lines(cfg_file: str) -> List:
    '''Read file lines to list and remove unnecessary characters like ‘\n’ and ‘#’.'''
    with open(cfg_file, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    return lines

def parse_cfg_list(cfg_list: List) -> List:
    '''Read attributes list and store them as key, value pairs in list blocks'''
    holder = {}
    blocks = []
    for cfg_item in cfg_list:
        if cfg_item[0] == '[':
            cfg_item = 'type=' + cfg_item[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = cfg_item.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks

def yolov3_net(cfg_file: str, model_size: int, num_classes: int):
    blocks = parse_cfg_file(cfg_file)
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    # normalize input to range of 0-1
    inputs = inputs / 255.0

    # YOLOv3 has 5 layers types:
    # 1. convolutional layer
    # 2. upsample layer
    # 3. route layer
    # 4. shortcut layer
    # 5. yolo layer
    for i, block in enumerate(blocks[1:]):

        # 1. convolutional layer
        if (block['type'] == 'convolutional'):
            activation = block['activation']
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            strides = int(block['stride'])
            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding='valid' if strides > 1 else 'same',
                            name=f'conv_{str(i)}',
                            use_bias=False if ('batch_normalize' in block) else True)(inputs)

            # there are 2 convolutional layer types, with and without batch normalization layer.
            # The convolutional layer with batch normalization layer uses a leaky ReLU activation layer,
            # otherwise, it uses the linear activation.
            if 'batch_normalize' in block:
                inputs = BatchNormalization(name=f'bnorm_{str(i)}')(inputs)
                inputs = LeakyReLU(alpha=0.1, name=f'leaky_{str(i)}')(inputs)

        # 2. Upsample Layer
        # Upsample by a factor of stride
        # e.g.: [upsample] stride=2
        elif (block['type'] == 'upsample'):
                stride = int(block['stride'])
                inputs = UpSampling2D(stride)(inputs)

        # 3. Route Layer
        # e.g.: [route] layers = -4
        #       Backward -4 number of layers, then output the feature map from that layer.
        # e.g.: [route] layers = -1, 61
        #       Concatenate the feature map from previous layer (-1) and the feature map from layer 61
        elif (block['type'] == 'route'):
            block['layers'] = block['layers'].split(',')
            start = int(block['layers'][0])

            if len(block['layers']) > 1:
                end = int(block['layers'][1]) - i
                filters = output_filters[i + start] + output_filters[i + end] # Index nagatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        # 4. Shortcut Layer
        # e.g.: [shortcut] from=-3 activation=linear
        #       Backward 3 layers (-3), then add the feature map with the feature map of the previous layer
        elif block['type'] == 'shortcut':
            from_ = int(block['from'])
            inputs = outputs[i - 1] + outputs[i + from_]

        # 5. Yolo Layer
        # Preform detection
        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = block['anchors'].split(',')
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            # reshape output to [None, B * grid size * grid size, 5 + C]
            # B = number of anchors, C = number of classes
            out_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])

            # access all boxes attributes
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]

            # refine bounding boxes prediction to right posisions and shapes.
            # sigmoid to convert to 0-1 range
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            # convert box_shapes
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

            # convert the relative positions of the  center boxes into the real positions
            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.shape[1] // out_shape[1], input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            # concatenate them all together
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

            # Yolov3 does 3 predictions across the scale.
            # Take prediction result of each scale and concatenate it with the others.
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1

            # Since the route and shortcut layers need output feature maps from previous layers,
            # we keep track of feature maps and output filters in every iteration.
        outputs[i] = inputs
        output_filters.append(filters)

    model = Model(input_image, out_pred)
    model.summary()
    return model
