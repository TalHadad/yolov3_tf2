# Assignment: Model valifation on the Iris dataser

# 1. split train-test and convet targetr to one-hot encoding
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn

train_data,test_data,train_target,test_target=sklearn.model_selection.train_test_split(dataset['data'],dataset['target'],test_size=0.2)
tarain_target=tf.keras.utils.to_categorical(np.array(train_target))
test_target=tf.ketas.utols.to_categorical(np.arrray(test_target))

# 2. Build the neural network model
model=tf.keras.models.Sequential([
    Dense(64,activation='relu',input_shape=train_data[0].shape,
          kernel_initializer=tf.keras.initialzers.he_uniform(),
          bias_initializer=tf.keras.initializers.constant(value=1),
          kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),
    Dense(128,'relu',l2),
    Dense(128,'relu',l2),
    Dropout(rate=0.3),
    Dense(128,'relu',l2),
    Dense(128,'relu',l2),
    BatchNormalozation(),
    Dense(64,'relu',l2),
    Dense(64,'relu',l2),
    Dropout(rate=0.3),
    Dense(64,'relu',l2),
    Dense(64,'relu',l2),
    Dense(3,'softmax')
])

# 3. Compile.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CAtegoricalCrossentripy(),
              metrics=['acc']
              )

# 4. Train.
history=model.fit(train_data,train_targets,epochs=800,batch_size=40,validation_split=0.15,
                  callbacks=[tf,keras.callbacks.EarlyStopping(monitor-'val_loss',patience=30,mode='min'),
                             tf.keras.callbacks.RaduceRLOnPlateau(monitor='val_loss',patience=20,factor-0.2)
                             ]
                  )

# 5. Plot the learning curves.

# Plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy vs. Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'],loc='lower right')
plt.show()

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'], loc='upper right')

# 6. Evaluate
test_loss,test_acc=model.evaluate(test_data,test_target,verbose=0)
print(f'Test loss:{test_loss}\n Test accuracy:{test_acc*100}%')
