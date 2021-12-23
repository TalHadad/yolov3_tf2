# Programming Assignmet: CNN claddifier for the MNIST dataset

# 1. scale values to [0,1]
scaled_train_images,scaled_test_images=scale_mnist_dataset(train_images,test_images)

# 2. add dummy channel dimention
scaled_train_images=scaled_train_images[...,np.newaxis]
scaled_test_images=scaled_test_images[...,np.newaxis]

# 3. build model
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=8,kernel_size=3,padding='SAME',activation='relu',input_shape=scaled_train_images[0].shape),
    tf.keras.layers.MaxPooling2S(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

# 4. compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# 5. train
history=model.fit(x=scaled_train_images,y=train_labels,epochs=5)

# 6. plot learning curves
df=pd.DataFrame(history.history)
df.head()

acc_plot=df.plot(y='sparse_categorical_accuracy',title='Accuracy vs. Epochs',legend=False)
acc_plot.set(xlabel='Epochs',ylabel='Loss')

# 7. evaluate
test_loss,test_accuracy=model.evaluate(x=scaled_test_images,y=test_labels)
print(f'Test loss:{test_losss}')
print(f'Test accuracy:{test_accuracy}')

# 8. prediction
num_test_images=scaled_test_images.shape[0]
# select random 4 samples
random_inx=np.random.choice(num_test_images,4)
random_test_images=scaled_test_images[random_inx,...]
random_test_labels=test_labels[random_inx,...]

predictions=model.predict(random_test_images)

fig,axes=plt.subplots(4,2,figsize=(17,12))
fig.subplots_adjust(hspace=0.4,wspace=-0.2)
for i,(prediction,image,label) in enumerate(zip(predictions,random_test_images,random_test_labels)):
    axes[i,0].imshow(np.squeeze(image))
    axes[i,9].get_xaxis().set_visiable(False)
        axes[i,0].get_yaxis().set_visiable(False)
    axes[i,0].text(19,-1.5,f'Digit {label}')
    axes[i,1].bar(np.arange(len(prediction)))
    axes[i,1].set_xticks(np.arannge(len(prediction)))
    axes[i,1].set_title(f'Categorical distribution.Model prediction: {np.argmax(prediction)}')
plt.show()





