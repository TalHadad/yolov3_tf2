# Assignment: Saving and loading models.

# 1. Return the filename of the latest saved checkpoint file.
callback=tf.keras.callbacks.ModelCheckpoint(filepath=f'checkpoint_every_epoch/checkpoint_{epoch}',save_only_weights=True, ...)
model.load_weights(filepath=tf.train.latest_checkpoint(checkpoint_dir='checkpoint_every_epoch'))