import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import matplotlib as plt

NAME = f'Cats-vs-dogs-cnn-64x2-{int(time.time())}'

tensorboard = TensorBoard(log_dir=f'tensorboard_logs/{NAME}')
# finally, from project folder path terminal run:
# >> cd /home/tal/Desktop/tf-lib/TensorFlow
# >> tensorboard --logdir='tensorboard_logs/'
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in=open("y.pickle","ed")
# y=pickle.load(pickle_in)

# X=X/255.

X=train_images
y=train_labels

model=Sequential([
	Conv2D(64,(3,3), input_shape=X.shape[1:]),
	Activation('relu'),
	MaxPooling2D(pool_size=(2,2)),

	Conv2D(75, (3,3)),
	Activation('relu'),
	MaxPooling2D(pool_size=(2,2)),

	Flatten(),

	Dense(64),
	Activation('relu'),

	Dense(1),
	Activation('sigmoid')
	])

model.compile(
	loss='binary_crossentropy',
	aptimizer='adam',
	metrics=['accuracy'])

def train():
	history=model.fit(X,y,
		batch_size=32,
		epochs=3,
		validation_split=0.3,
		callbacks=[tensorboard])
	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

	plt.figure(figsize=(10,10))
	for i in range(25):
	    plt.subplot(5,5,i+1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(train_images[i])
	    # The CIFAR labels happen to be arrays, 
	    # which is why you need the extra index
	    plt.xlabel(class_names[train_labels[i][0]])
	plt.show()

if __name__=='__main__':
	train()