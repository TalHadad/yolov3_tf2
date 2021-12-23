###################################
# 1. Model validation
###################################

# 1.1.
model.fit(input,targets,validation_split=0.2)

# 1.2.
history=model.fit(...,validation_split=0.2)
print(history.history.keys())
# dict_keys(['loss','mape' (-> for training)
#            'val_loss','val_mape' (-> for validation) ])

# 1.3.
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
model.fit(x_train,y_train,validation_data=(x_test,y_test)) # explicitly give the validation set.

# 1.4.
# making training and validation split ourself.
# * test_size=0.2 and validaton_split=0.25 split the data to 60(train)/20(val)/20(test)
for sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1)
model.fit(...,validation_data=(x_val,y_val))

# 1.5.
# overfiting: *graph of loss over epochs: loss is reducing in training, but incresseing in validation*,
# next we will see how to reduce overfiting.

###################################
# 2. Model regularisation.
###################################

# 2.1.
model=Sequential([
    Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(1,activation='sigmoid')
])
#'weight decay' is same as 'L2 regularization'.
# 0.001=the coefficient (that multiplies the sum of squared weights in this layer).
# weight matrix is the kernel.

model.compile(...,adadelta,
              binary_crossentropy,
              acc)
model.fit(...,validation_split=0.25)
# * L2 Regularization is adding something to the loss function:
# penalizing large values of the weights, encourages simpler function (smooth).
# L(y_pred,y_true)=SUM{from i=1 to train.shape[0]}((y_pred-y_true)^2) (-> binary_crossentropy) +
#                  GAMMA (-> weight decay/coefficient of the layer) *
#                  SUM{from i=1 to all weights in the layer}((W_i)^2)

# 2.2.
kernel_regularizer=tf.keras.regularizers.l1(0.005)
# * L1 Regulatization:
# encorage sparse weights (some values turned into zeros).
# L(y_pred,y_true) = SUM{from i=1 tp train.shape[0]}((y_pred-y_true)^2) (-> binary_crossentropy) +
# GAMMA * SUM{from i=1 to all weights of the later}(|W_i|) (-> l1 regularization)

# 2.3.
kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.005,l2=0.001)
# L(y_ppred,y_true)=SUM{from i=1 to train.shape[0]}((y_pred-y_true)^2) +
#                   0.005 (-> l1) * SUM{from i=1 to layer wughts}(|W_i|^2) +
#                   0.001 (-> l2) * SUM{grom i=1 to layrt wights}(W_i^2)

# 2.4.
# If multiple ;ayers have regularizers than the loss function will look like:
# L(y_pred,y_true)= SUM((y_pred-y_true)^2) (-> binary_crossentropy) +
#                   l1_layer1 * SUM{layer1}(|W|) + l2_layer1 * SUM{layer1}(W^2) (-> regularize layer 1) +
#                   l1_layer2 * SUM{layer2}(|W|) + l2_layer2 * SUM{layer2}(W^2) (-> regularize layer 2) + ...

# 2.5.
# regularization on the bias of the layer.
Dense(...,bias_regularizer=tf.keras.regularizers.l2(0.001))

# 2.6.
# Dropouts also has a regularizing effect.
# 0.5 = dropout rate, the probability that a value will set to zero.
# * Its Bernoulli Dropout, since the weights are effectively being multiplied by a Bernoulli random variable.
model=Sequential([
    Dense(...),
    Dropout(0.5),
    Dense(...), ...])

model.compile(...)
# Training mode, with dropout
model.fit(...)
# Testing mode, no dropout
model.evaluate(...)
model.puedict(...)
# * Note: In TensorFlow 1 the dropout rate was 1-p (p = was the keep rate).

###################################
# 3. Batch Normalization.
###################################

from tensorflow.keras.layers import BatchNormalization,...

# 3.1.
model=Sequential([
    Dense(64, input_shape=[train_data.shape[1],], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(),
    Dense(256,activation='relu')
])

# 3.2.
model.add(tf.kerad.layers.VatchNormalizaton(
    momentum=0.99, # weighting given to the previous running mean when re-computing it with a extra minibatch.
    epsilon=0.001, # numeric stability.
    axis=-1, # the axis gor batch normalization.
    bera_initializer=tf.keras.initializers.RandomNorm(mean=0.0,stddev=0.05), # affine transformation after normalization.
    gamma_initializer=tf.keras.initializers.Constant(value=0.0)) # affine ... (the previous comment is for both lines).
)
model.add(Dense(1)) # output layer.
model.compile(...)
model.fit(...)
df.plot(history...)

###################################
# 4. Callbacks.
###################################

from tensorflow.keras.callback import Callback

# 4.1.
class my_callback(Callback):
    def on_train_begin(self, logs=None):
        # Do something at the start of training
        pass
    def on_train_batch_begin(self, batch,logs=None): # batch = batch number
        # Do something at the start of every batch iteration
        pass
    def on_epoch_end(self, epoch, logs=None): # epoch = epoch number
        # Do something at the end of every epoch
        pass

# 4.2.
model.fit(x_train,y_train,epochs=5,callbacks=[my_callback()])

# 4.3.
on_train_... # methods will be called when you run model.fit()
on_test_... # methods will be called by model.evaluate()
on_predict_... # methods will be called by model.predict()
# * history=model.fit(), history is also a callback that saves in a dictionary the values of loss and metrics in each epoch.

# 4.4.
# used when fitting the model: model.fit(..., callbacks=[TrainingCallback()])
class TrainingCallback(Callback):
    def on_train_begin(self, loss=None): ...
    def on_epoch_begin(self, epoch, logs=None): ...
    def on_train_batch_begin(self, batch, logs=None): ...
    def on_train_batch_end(self, batch, logs=None): ...
    def on_epoch_end(self, epoch, logs=None): ...
    def on_train_end(self, logs=None): ...

# 4.5.
# used when evaluating the model: model.evaluate(..., callbacks=[TestingCallback()])
class TestingCallback(Callback):
    def on_test_begin(self, logs=None): ...
    def on_test_batch_begin(self, batch, logs=None): ...
    def on_test_batch_end(self, batch, logs=None): ...
    def on_test_end(self, logs=None): ...

# 4.6.
# used when predicting by the model: model.predict(..., callbacks=[PredictionCallback()])
class PredivtionCallback(Callback):
    def on_prediction_begin(self, logs=None): ...
    def on_prediction_batch_begin(self, batch, logs=None): ...
    def on_prediction_batch_end(self, batch, logs=None): ...
    def on_prediction_end(self, logs=None): ...

###################################
# 5. The log dictionary.
###################################

# 5.1.
# print the loss and metrics while training and evaluating
def on_reain_batch_end(self, batch,logs=None):
    print(f'batch {batch} loss: {logs["loss"]}')
def on_test_batch_end(self, batch, logs=None):
    print(f'batch {batch} loss: {logs["loss"]}')
def on_epoch_end(self, epoch, loss=None):
    print(f'epoch {epoch} loss: {logs["loss"]}, mea: {logs["mae"]}')

# 5.2.
# Define a callback to cahand the learning rate of the optininzer during trainging
lr_schedule=[(4,0.03),(7,0.02),(11,0.005),(15,0.007)]
def get_new_epoch_lr(epoch,lr):
    for e_s,l_s in lr_schedule:
        if e_s==int(epoch):
            return l_s
        return lr

class LRScheduler(Callback):
    def __init__(self, new_lr_function):
        super(LRScheduler,self).__init__()
        self.new_lr_function=new_lr_function
    def on_epoch_begin(self,epoch,logs=None):
        if not hasattr(self.model.optimizer,'lr'):
            raise ValueError('Error: Optimizer dont have learning rate')
        curr_rate=float(tf.keras.backend_get_value(self.model.optimizer.lr))
        scheduled_rate=self.new_lr_function(epoch,curr_rate)
        tf.keras.backend.set_value(self.model.optimizer.lr,scheduled_rate)

model=Sequential(...)
model.compile(...)
history=model.fit(..., callbacks=[LRScheduler(get_new_epoch_lr)])

#################################################3
# 6. Early stopping and patience
#################################################

from tensorflow.keras.callbacks import EarlyStopping

# 6.1.
early_stopping=EarlyStopping()
model.fit(..., balidation_s[lot=0.2, callbacks=[early+stopping])

# 6.2.
# Specifing on which performance the callvack is monitoring (the degault id 'val_loss')
# if mereics=['accuracy'] then you can use 'cal_accuracy' (same sting as the metric)
early_stopping=EarlyStopping(monitou='cal_loss')

# 6.3.
# training stops if there is no improvement for 5 (or patience) epochs in a row (default is zero)
early_stopping=EarlyStoping(monitor='val_loss',patience=5)

# 6.4.
# define what qualifies as an impronment (default is zero)
early_stoppping=EarlyStopping(..., min_swlta=0.01)

# 6.5.
# an improvement could be either an increase ('max') or a decrease ('min') in the quantity that we are monitoring.
# Validation loss or error should go down ('min'), but validation accutacy shoud go up ('max).
# The defoult id 'auto', the direction id ingerred by the quanitity name.
early_stopping=EarlyStopping(..., mode='max')

########################################
# 7. Additional callbacks.
######################################

# 7.1.
# changing the learning rate while training using custom function
def lr_function(epoch,lr):
    if epoch%2==0:
        return lr
    else:
        return lr + epoch/1000

history=model.fit(...,
                  callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_function,verbose=1)]
                  )

# 7.2.
# changing the learning rate while training for each epoch using lambda function.
tf.keras.callbacks.LearningRateScheduler(lambda x: 1/(3+5*x), verbose=1)

# 7.4.
# quickly define simple custom callbacks with lambda functions
history=model.fit(...,
                  callbacks=[tf.keras.callbacks.LambdaCallback(
                      on_epoch_begin=(lambda epoch, logs: print(...)),
                      on_batch_end=(lambda batch, logs: print(...)),
                      on_train_end=(lambda logs: print(...)))]
                  )

# 7.5.
# Reducing the learning rate when a metric has stopped improving: new_lr=factor*old_lr
# The arguments are similar to EarlyStoppin.
history=model.fit(...,
                  callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1)]
                  )







