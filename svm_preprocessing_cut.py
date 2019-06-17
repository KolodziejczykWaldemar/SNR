from keras import applications
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
import numpy as np
from keras import metrics
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pandas as pd
import sklearn.metrics as skm

img_width = 128
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 1
class_nb = 120

base_model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_width, 3))

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

for layer in classifier.layers:
    print(layer, layer.trainable)
print(classifier.summary())

model_type = 'cut_layer'
classifier.load_weights("models/weights_{}.best.hdf5".format(model_type))


final_model = Sequential()
for layer in classifier.layers[:-1]:
    final_model.add(layer)

final_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

print(final_model.summary())


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
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(15780):
    _x, _y = train_set.next()
    _x_embed = final_model.predict(_x[:, :, :])
    X_train.append(_x_embed[0, :])
    y_train.append(_y[0, :])
    if i % 100 == 0:
        print(i)

for i in range(3600):
    _x, _y = valid_set.next()
    _x_embed = final_model.predict(_x[:, :, :])
    X_test.append(_x_embed[0, :])
    y_test.append(_y[0, :])
    if i % 100 == 0:
        print(i)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train_df = pd.DataFrame(X_train)
X_train_df.to_csv('SVM_cut_X_train.csv', index=False)

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv('SVM_cut_y_train.csv', index=False)

X_test_df = pd.DataFrame(X_test)
X_test_df.to_csv('SVM_cut_X_test.csv', index=False)

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv('SVM_cut_y_test.csv', index=False)
