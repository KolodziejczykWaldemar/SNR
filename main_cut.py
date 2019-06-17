from keras import applications
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from keras import metrics
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pandas as pd

img_width = 128
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 32
class_nb = 120

base_model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_width, 3))


# Compiling the CNN

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
# for layer in model.layers[:-3]:
#     layer.trainable = False
#
# for i in range(len(model.layers)):
#     print(i, model.layers[i], model.layers[i].trainable)

model = Sequential()
for layer in base_model.layers[:50]:
    model.add(layer)

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(class_nb, activation="softmax")(x)

# creating the final model
classifier = Model(input=model.input, output=predictions)
print(classifier.summary())

for layer in classifier.layers:
    print(layer, layer.trainable)


classifier.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_set = train_datagen.flow_from_directory('dataset/train_set',
                                              target_size=(img_width, img_width),
                                              batch_size=batch_size,
                                              class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_set = valid_datagen.flow_from_directory('dataset/valid_set',
                                              target_size=(img_width, img_width),
                                              batch_size=batch_size,
                                              class_mode='categorical')

checkpoint = ModelCheckpoint('models/weights_cut_layer.best.hdf5', monitor="acc", verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]

history = classifier.fit_generator(train_set,
                                   steps_per_epoch=15780//batch_size,
                                   nb_epoch=50,
                                   validation_data=valid_set,
                                   nb_val_samples=3600,
                                   callbacks=callbacks_list,
                                   verbose=1)

model_json = classifier.to_json()
with open("models/mobilenet_120_dogs.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("models/mobilenet_120_dogs.h5")


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

df = pd.DataFrame({'acc': acc, 'val_scc': val_acc, 'loss': loss, 'val_loss': val_loss})
df.to_csv('history_cut_layer.csv')

epochs = range(len(acc))

plt.figure(figsize=(15, 8))
plt.plot(epochs, acc, color='#6ee052', label='Training acc')
plt.plot(epochs, val_acc, color='#6052e0', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('history_acc_cut_layer.png')

plt.figure(figsize=(15, 8))
plt.plot(epochs, loss, color='#6ee052', label='Training loss')
plt.plot(epochs, val_loss, color='#6052e0', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('history_loss_cut_layer.png')
