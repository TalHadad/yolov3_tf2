###########################################
# 1. Saving and loading model weights.
###########################################

# You can save:
#   1. model parameters (weights)
#   2. entire model (weights and architecture)
#   3. model architecture

# 1.1. Saving model weights:
from tensorflow.keras.callbacks import ModelCheckpint
model=Sequential([...])
model.compile(...)
checkpoint=ModelCheckpoint('my_model',save_weights_only=True)
model.fit(..., callbacks=[checkpoint])
# 3 files are created in the currtnt working directory:
# (TensorFlow format)
#   1. checkpoint
#   2. my_model.data-00000-of-00001
#   3. my_model.index

# 1.2.
checkpoint=ModelCheckpoint('keras_model.h5',save_weights_only=True)
# a single file is created in the current working directory:
# (Keras format)
#   1. keras_model.h5

# For most models it doesn't matter which format you use,
# but in general it recommended to use the native TensorFlow format.

# 1.3.
# load the weights after theyre initialized.
# * you need to know the model achitecture and build the model.
model=Sequential([...])
model.load_weights('my_model')

# 1.4.
model=Sequential([...])
model.load_weights('keras_model.h5')

# 1.5.
# Saving the weights manually (without ModelCheckpoint),
model=Sequential([...])
model.comppile(...)
model.fit(...)
model.save_weights('my_mode')

# 1.6.
# for every epoch we overwrite the same checkpoint file.
checkpoint=ModelCheckpoint(filepath='model_checkpoint/checkpoint',frequency='epoch',save_weight_only=True,verbose=1)

# 1.7.
# clear the model_checkpoints directory
! run -r model_checkpoints

################################################
# 2. Explanation of saved files.
#######################################

# The ModelCheckpoint or save_weights create three files:
#   1. checkpoint (~87B): metadata that indicates where the actual model data is stored.
#       model_checkpoint_path:'checkpoint'
#       all_model_checkpoint:'checkpoint'
#   2. checkpoint.index: which weights are stored where
#       (in distributed systems,the model may have to be recomposed from multiple shards)
#   3. checkpoint.data-00000-of-00001: contains the actual weights of the model.

#########################################
# 3. Model saving criteria.
########################################

# 3.1.
# Save weights for each epoch (default).
# Save_freq=1000, the number of samples that been seen by the model since the last time the weights were saved
# (number of samples and not training iterations)
checkpoint=ModelCheckpoint('training_run_1/my_model',save_weights_only=True,save_freq='epoch') # (same as frequency?)

# 3.2.
# Only save the weight if the monitored measure (default 'val_loss') is the best value seen so far in the training run
# (similar to early stopping)
checkpoint=ModelCheckpoint(..., save_best_only=True) # (default is False)

# 3.3.
# Same as early stopping, specify if to minimize or maximize the monitored performance measure.
checkpoint=ModelCheckpoint(..., monitor='val_acc', mode='max') # (default is auto)

# 3.4.
# The file name can be formatted with whatever keys are available in the logs dictionary,
# e.g. 'my_model.{epoch}-{val_loss:.4f}'.
# The files will not be overwritten.
checkpoint=ModelCheckpoint('training_run_1/my_model.{epoch}.{batch}', ...)

#########################################
# 4. Saving the entire model.
###########################################

# 4.1.
# Saving the entire model (and not just the weights).
checkpoint=ModelCheckpoint('my_model', save_weights_only=False) # default is False.
# 3 files and 2 folders are created in the working directory:
#   1. my_model/assets
#   2. my_model/saved_model.pb
#   3. my_model/variables/variables.data-00000-of-00001
#   4. my_model/variables/variables.index
# * The file path is used to create a subdirectory (my_model/...)
#   (when saving only weights, the filepath is the files name).
# * Two more subdirectories are created:
#   1. my_model/assents/: (1.) where files can be stored that are used by the TensorFlow graph.
#   2. my_model/variables/: (3., 4.) contains the saved weights of the model
#      (same the file types as when saving weights only).
#  * And the last file:
#   3. my_model/saved_model.pd: (2.) the file that stores the TensorFlow graph itself
#      (model architecture - build, compaile, optimize).

# 4.2.
checkpoint=ModelCheckpoint('keras_model.h5',save_weights_only=False)
# Just one file will be saved:
#   1. keras_model.h5 (same as saving weights only,
#      but this HDF5 file now contains the architecture as well as the weights).

# 4.3.
# Manually save the weights of the model after training (save Model format),
# for weights only we used save_weights.
# Save the same as naive TensorFlow format.
model-Sequential([...])
model.compile(...)
model.fit(...)
model.save('my_model')

# 4.4.
# HDF5 format.
model.save('keras_model.h5')

# 4.5.
# returns the complite model instance.
from tensorflow.keras.models import load_model
new_model=load_model('my_model')
new_model.fit(...)
new_model.evaluate(...)
new_model.predict(...)

# 4.6.
# * Loading the entire model id slower than loading weights only.
# * The files sizer are bigger for the entire model then weights only.
# * Explore the differences between TensorFlow's .pd format and Keras' .h5 format
new_keras_model=load_model('keras_model.h5')

#################################################3
# 5. Saving model architecture only.
#################################################

# 5.1.
# dictionary of all module build parameters(architecture)
config_dict=model.get_config()
print(config_dict)

# 5.2.
# Return model with the same layers, but not the same initialized weights values.
model_same_config=tf.keras.Sequential.from_config(config_dict)
model.get_config()==model_same_config.get_config() # true
np.allclose(model.weights[0].numpy(),model_same_config.weights[0].numpy()) # false

# 5.3.
# For models that are not sequentials.
model_same_config=tf.keras.Model.from_config(config_dict)

# 5.4.
# Json format
json_config=model.to_json()
with open('config.json','w') as f:
    json.dump(json_config,f)
del json_config

# 5.5.
# Json format
with open('config.json','r') as f:
    json_config=json.load(f)
model_same_config=tf.keras.model_from_json(json_config)

# 5.6.
# Yaml format (same as json)
yaml_config=model.to_yaml()
with open('config.yaml','w') as f:
    yaml.dump(yaml_config,f)
del yaml_config

# 5.7.
# Yaml format (same as json)
with open('config.yaml','r') as f:
    yaml_config=yaml.load(f)
model_same_config=tf.keras.model_from_yaml(yaml_config)


###########################################3
# 6. Loading pre-trained Keras models.
###########################################

# * Download from keras.io/applications you'll see a number of different architectures:
# Xception, VGG16, VGG19, ResNet, Inception, MobileNet, DenseNet, NASNet, ...
# with examples of how to use as classifier of feature extractor.

# 6.1.
# The architecture and weights will be download (if not exist) to a hidden golder ~/.keras/models/.
# 'imagenet' are the weights learned on ImageNet datader.
# weights=None for randomly initialization.
from tensorflow.keras.application.resnet50 import ResNet50
model=ResNet50(weight='imagenet')

# 6.2.
# include_top=False means that the fully connected layer at the top of the network isn't loaded (headless model),
# for transfer learning applications.
model=ResNet50(weights='imagenet', include_top=False) # the default is True,
# which means that the complete classifier model is downloaded.

# 6.3.
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
img_input=image.load_img('my_picture.jpg', target_size=(224,224))
img_input=preprocess_input(img_input[np.newaxis, ...])

# 6.4.
# List of (class, description, probability).
from tensorflow.keras.applications.resnet50 import decode_predictions
preds=model.predict(img_input) # numpy array of probabilities
decoded_predictions=decode_predictions(preds, top=3)[0] # List of (class, description, probability).
# top=3 the top three predictions

##############################
# 7. TensorFlow Hub modules.
##############################

# TensorFlow Hub id another resource for pre-trained models, more focused on network modules
# (separate components of an overall TensorFlow graph).
# Go to tensorglow.org/hub
# Install using ! pip install 'tensorflow_hub>=0.6.0'

# 7.1.
import tensorflow_hub as hub
module_url='https://tfgub.dev/google/imagenet/mobilenet_v1_050_160/classification/4'
model=Sequential([hub,KerasLayer(module_url)])
model.build(input_shape=[None,160,160,3])

# 7.2.
module=tf.keras.models.load_model('models/Tensorflow_MobileNet_v1')
model=tf.keras.models.Sequential(hub.KerasLayer(module))
model.build(input_shape=[None,160,160,3])

# 7.3.
# changes the ImageNet categories from a numeric value {0,1,...,999} to human-readable format.
# (provided at tensorflow.org/hub)
lemon_img=tf.keras.preprocessing.image.load_img('data/lemon.jpg',target_size=(160,160))
with open('data/imagenet_categories.txt') as txt_file:
    categories=txt_file.read().splitlines()


