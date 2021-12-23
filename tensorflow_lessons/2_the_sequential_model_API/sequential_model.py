# 1. Introduction - The sequential model API

# The keras progect was authoued by Francois Chollet
# tf.keras focumentation is found in keras.io and tensorflow.org->APU->tf.keras (prefered)

###################################
# 1. Feedforward neural networks.
###################################

# 1.1.
from tensorlow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequntial([Dense(64,activation='relu'), Dense(10,activation='softmax')])
#64 neurons (units)
#if activation is not mention, the default is linear or no actibation.
# if input_shape is not mention, the weights and biases not yet created

# 1.2.
Dense(64,actibarion='relu',input_shape=(784,))
# the weight and biases will be created and initialized straight away.

# 1.3.
# alternative way to build the same model.
model=Sequential()
model.add(Dense(64,activation='relu',input_shap=(784,)))
model.add(Dense(10,activation='softmax'))

# 1.4.
from tensorflow.keras.layers import Flatten
model=Sequential([Flatten(input_shape=(28,28)),  #(784,)
                  Dense(64,activation='relu'),
                  Dense(10,actvation='softmax')])

# 1.5.
# the values and shapes of all layers weights
model.weight

# 1.6.
# a nicer way to see the layers shapes
model.summary()

# 1.7.
# equal to Sense(10,activation='softmax'), because the default activation id linear
Dense(10)
Softmax()

# 1.8.
# the name will appear in the summary
Dense(16, activation='relu',name='layer_1')

###################################
# 2. Convolutional neural networks.
###################################

# 2.1.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D

model=Swqurntial([Conv2D(16,(3,3),activation='relu',input_ghape=(32,32,3)), # 16 filters, (3,3) shape of filter
                  # (None,30,30,16) None=batch size, 30=(((32+2*0)-3)/1)+1)=(((in+2*padding)-kernel)/stribe), 16=filters
                  MaxPooling2D((3,3)), # (3,3) shape of pooling window
                  # (None,10,10,16) 10=((30-3)/3)+1
                  Flatten(),
                  # (None,1600) 1600=10*10*16
                  Dense(64,activation='relu'),
                  # (None,64) 64=units (or neurons)
                  Dense(10,activation='softmax')])
                  # (None,10) 10=units (or neurons)

# 2.2.
# padding = 'Full'(=k-1)/'SAME'(=(k-1)/2)/'VALID'(=0)
Conv2D(16,kernal_size=3, padding='SAME', ...)
MaxPooling2D(pool_size=3)

# 2.3.
# kernel_size and pool_size=3 is a shortcut for (3,3)

# 2.4.
Conv2D(16,(3,3),activation='relu', input_shape=(28,28,1), data_format='channel_last')
# 'channel_last' (default) means that the input_shape=(28,28,1) 1=channel
# 'channel_first' means input_shape=(1,28,28) 1=channel

###################################
# 3. Weight initialisation.
###################################

# 3.1.
# Default for Dense: weights in range [-sqr(6/(n_input+n_output)), sqr(6/(n_input+n_output))]
# (Glorot uniform initializer = glorot_uniform)
# biases=0 (zeros)

# 3.2.
# Initializing weights and biases:
Conv1D(filters=16,kernel_size=3,input_shape=(128,64),kernel_initializer='random_uniform',bias_initializer='zeros', activation='relu')
# ...
Dense(64,kernel_initializer='he_uniform',bias_initializer='ones',activation='relu')

# * if a layer has no weighs of biases (such as MaxPooling), then error is thrown when setting kernel of bias initializer.

# * if you want to change the default paraneters of initializer use the long version:
kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05)
# or tf.keras.initializers.Orthogonal(gain=1.0,seed=None)
bias_initializer=tf.keras.initializers.Constant(value=0.4)

# 3.3.
# Custom weight and bias initializers:
import tensorflow.keras.backend as K
def my_init(shape,dtype=None):
    return K.random_normal(shape,dtype=dtype)
model.add(Dense(64,kernel_initializer=my_init))
# Initializers must take two arguments:'shape','dtype' of the input tensor,
# and return whatever K.random_normal return

# 3.4.
# Plot histogram of the initialized values:
import matplotlob.pyplot as plt
fig,axes=plt.subplots(5,2,figsize=(12,16)) # 5 rows, 2 columns
fig.subplots_adjust(hspace=0.5,wspace=0.5)
weight_layers=[layer for layer in model.layers if len(layer.weights)>0] # filter out pooling and flatten layers without weights
for i,layer in enumerate(weight_layers):
    for j in [0,1]:
        axes[i,j].hist(layer.weights[j].numpy().flatten(),align='left')
        axes[i,j].set_title(layer.weights[j].name)

###################################
# 4. Compiling your model.
###################################

# 4.1.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(64,activation='elu', input_shape=(32,)), # exponential linear unit activation function
                    Dense(1,activation='sigmoid')])
model.compile(optimizer='sgd', # stocastic gradient decent
              loss='binary_crossentropy', # 'binary_crossentropy' makes sense for the network and task
              metrics=['accuracy','mae']) # calculated for each epoch.
# * optimizer = 'sgd'/'adam'/'rmsprop'/'adadelta'
# * loss = 'binary_crossentropy'/'mean_squared_error' (for regression task)/'categorical_crossentropy'/
#  'sparse_categorial_crossentropy' (for multi-class classification task with sparse labels)

# 4.2.
# if you want to change the default parameters of optimizer, loss and metrics.
model.compaile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,mumentum=0.9,nesteron=True),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # if output activation os 'linear'(default)
    # * more numericaly stable approach to use linear
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7),
             tf.keras.metrics.MeanAbsoluteError()]
)

# 4.3.
print(model.optinizer) # or model.optimizer.lr for the learning rate
print(model.loss)
print(model.metrics)

###################################
# 5. optimizers, loss finctiond and metrics.
###################################

# 5.1.
# Case 1: Binary Classification (with sigmoid or softmax):
# x_i -> linear -> sigmoid (m=1 classes, [0,1]) or softmax (m=2 classes, [[0,1],[0,1]]) -> round (threshold=0.5) = y_pred_i from {0,1} (0 or 1)
# accuracy is the same when using sigmoid or softmax because its binary classification
# accuracy = 1/N sum[i=1 to N](1 if y_pred_i=y_true_i, else 0)
# accuracy = K.mean(K.equal(y_true,K.round(y_pred)))

# 5.2.
# Case 2: Categorical Classification:
# x_i -> linear -> softmax (m>2 classes, [[0,1],[0,1],...,[0,1]]) -> index of max value = y_pred_i from {0,1,...,m-1}
# accuracy = 1/N sum[i=1 to N](1 if y_pred_i=y_true_i, else 0)
# accuracy = K.mean(K.equal(K.argmax(y_true,axis=-1), K.argmax(y_pred,axis=-1)))

# 5.3.
# Binary Accuracy: change default threshold (=0.5)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
# instead of using metrics=['binary_accuracy']

# 5.4.
# Sparse Categorical Accuracy: y_true=integer (class num), and not one-hot vector
model.compile(opt=..., loss=..., metric=['sparse_categorical_accuracy'])
# or tf.keras.metrics.SparseCategoricalAccuracy()

# 5.5.
# Top K Categorical Accuracy: y_true is in the k highest index
model.compile(opt=..., loss=..., metrics=['top_k_categorical_accuracy'])
# or tf.keras.metrics.TopKCategoricalAccuracy(k=5)

# 5.6.
# Sparse Top K Categorical Accuracy: y_true=integer and is in the highest indexs
model.compile(opt=..., loss=..., metrics=['sparse_top_k_categorical_accuracy'])
# or tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

# 5.7.
# Custom metrics:
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)
model.compile(opt=..., loss=..., metrics=[mean_pred])

# 5.8.
# Multiple metrics:
model.compile(opt=..., loss=..., metrics=[mean_pred,'accuracy',tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])

###################################
# 6. Training your model.
###################################

# 6.1.
model=Sequential([Dense(64,activation='elu',input_shape=(32,)),
                  Dense(100,activation='softmax')])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=16)
# assuming that x and y are numpy arrays and
# x_train:(num_sample,num_features),
# y_train:(num_sample,num_classes) num_classes is one-hot encoding (0 and 1 at class index)
# or y_train:(num_sample,) sparse representation (integer, the class index) (have to use loss='sparse_...')
# default of batch size is 32

# 6.2.
history=model.fit(...) # records the loss and metrics for each epoch

# 6.3.
# test and normolize the datasets
train_images.shape #(60000,28,28)
train_labels[0] # 9 (class 9)
train_images=train_images/255.
test_images+test_images/255.

# 6.4.
# print example image and label.
image_index=0
img=train_images[image_index,:,:]
plt.imshow(img)
plt.show()
print(f'label: {labels[train_labels[i]]}')

# 6.5.
model.fit(train_images,train_labels,epochs=2,batch_size=256) # return error!
# to fix the dimentions (adding a dummy channel dimention) use:
# train_images[...,np.newaxis]

# 6.6.
model.fit(...,verbose=2)
# verbose = {
#   0 (don't print progress),
#   1 (print bynamically per iteration),
#   2 (ptint one line per epoch)
# }

# 6.7.
# pandas summary of training progress
df=pd.DataFrame(history.history)
df.head()

# 6.8.
# plot thr loss while training.
# to print the mesures:
#   accuracy use 'Sparse_categorical_accuracy',
#   mae use 'mean_absolute_error'
loss_plot=df.plot(y='loss',title='Loss vd. Epochs',legend=False)
loss_plot.set(xlabel='Epochs',ylabel='Loss')

###################################
# 7. Evaluating and prediction.
###################################

# 7.1.
model=Sequential(...)
model.compile(...)
model.fit(...)
mjodel.evaluate(x_test,y_test) # calculate the loss and metrics on the test set.

# 7.2.
# save the value of evaluation (if there were more metrics, they'll all be retured by evaluate())
loss,accuracy,mae=model.evaluate(...)

# 7.3.
# x_sample is a numpy array shape: (num_samples,num_features), input_shape=(num_features,).
# return the outputs of the network for each sample.
# for a single example shape will be (1,num_feateues) and the output (1,num_classes), e.g. [[0.07]] (1 class) (2D),
# for two examples, e.g. (2,12), return [[0.07],[0.94]] (2D).
pred=model.predict(x_sample)

# 7.4.
# add dummy channel, and for prediction ude in the beggining and the end.
model.evaluate(test_image[...,np.newaxis], test_labels,vetbose=2)

# 7.5.
model.predict(test_image[np.newaxis,...,np.newaxis]]

# 7.6.
# select the highest class probability
predictions=model.predict(...)
print(f'Model prediction:{labels[np.argmax(predictions)]}')


















