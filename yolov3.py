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

def parse_cfg_file(cfg_file:str):
    lines = read_uncommented_lines(cfg_file)
    blocks = parse_cfg_list(lines)
    return blocks

def read_uncommented_lines(cfg_file:str) -> List:
    with open(cfg_file, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    return lines

def parse_cfg_list(cfg_list:List) -> List:
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
