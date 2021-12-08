#convert_weights.py
'''
When we re-write the original YOLOv3 weights (binary file, type float) to TensorFlow’s format
for a convolutional with a batch normalization layer,
we need to switch the position of beta and gamma.
So, they’re ordered like this: beta, gamma, means, variance and conv weights.
However, weights’ order remains the same for the convolutional without a batch normalization layer.
(Original sturctue is:
Convolutional with batch normalization: gamma, beta, mean, variance, conv weights
Without: conv biases, conv weights)
'''
import numpy as np
from yolov3 import yolov3_net
from yolov3 import parse_cfg_file

def load_weights(model, cfg_file, weight_file):
    '''read weights file and convert them to tensorflow2 weights'''
    fp = open(weight_file, "rb")

    # skip header (first 5 lines)
    np.fromfile(fp, dtype=np.int32, count=5)

    blocks = parse_cfg_file(cfg_file)

    # iterate blocks to know if the convolutional layer is with batch normalization or not.
    for i, block in enumerate(blocks[1:]):
        if (block['type'] == 'convolutional'):
            conv_layer = model.get_layer('conv_' + str(i))
            print('layer: ', i+1, conv_layer)

            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if 'batch_normalize' in block:
                norm_layer = model.get_layer('bnorm_' + str(i))
                print('layer: ', i+1, norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)

                bn_weights = np.fromfile(fp, dtype=np.float32, count=4*filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if 'batch_normalize' in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        # Alert if the reading has failed
    assert len(fp.read()) == 0, 'failed to read all data'

    fp.close()

def main():
    weight_file = 'weights/yolov3.weights'
    cfg_file = 'cfg/yolov3.cfg'

    model_size = (416, 416, 3)
    num_classes = 80

    model = yolov3_net(cfg_file, model_size, num_classes)
    load_weights(model, cfg_file, weight_file)

    try:
        model.save_weights('weights/yolov3_weights.tf')
        print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print('Couldn\'t write the file \'yolov3_weights.tf\'.')

if __name__ == '__main__':
    main()

# Execute on bash:
#    $ python convert_weights.py
# The command create 4 new file, TensorFlow 2.0 weights format:
#    1. checkpoint
#    2. yolov3_weights.tf.data-00000-of-00002
#    3. yolov3_weights.tf.data-00001-of-00002
#    4. yolov3_weights.tf.index
# To use them, call them as one file: yolov3_weights.tf
