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


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


img_width = 128
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 1
class_nb = 120

model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_width, 3))


for layer in model.layers:
    print(layer, layer.trainable)

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(class_nb, activation="softmax")(x)

# creating the final model
classifier = Model(input=model.input, output=predictions)
print(classifier.summary())

model_type = 'all_layers'
classifier.load_weights("models/weights_{}.best.hdf5".format(model_type))
# Compile model (required to make predictions)
classifier.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
print("Created model and loaded weights from file")
# load pima indians dataset

valid_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_set = valid_datagen.flow_from_directory('dataset/valid_set',
                                              target_size=(img_width, img_width),
                                              batch_size=batch_size,
                                              class_mode='categorical')


# scores = classifier.evaluate_generator(valid_set,
#                                        steps=3600,
#                                        verbose=1)
# print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
# print(scores)
import numpy as np

X_test = list()
y_test = list()
y_score = list()

for i in range(3600):
    _x, _y = valid_set.next()
    X_test.append(_x)
    y_test.append(_y[0, :])
    _y_hat = classifier.predict(_x[:, :, :])
    y_score.append(_y_hat[0, :])
    if i % 100 == 0:
        print(i)

X_test = np.array(X_test)
print(X_test.shape)

y_test = np.array(y_test)
print(y_test.shape)

y_score = np.array(y_score)
print(y_score.shape)

class_indices = np.arange(class_nb)
print(y_test.dot(class_indices).astype(int))
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as skm

# Plot linewidth.
lw = 2
n_classes=class_nb
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
accuracy = []
precision = []
f1 = []
recall = []
TP = []
TN = []
FP = []
FN = []

y_score_bin = []
for i in range(3600):
    zer = np.zeros(n_classes)
    idx = np.argmax(y_score[i, :])
    zer[idx] = 1
    zer = zer.astype(int)
    y_score_bin.append(zer)

y_score_bin = np.array(y_score_bin)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    accuracy.append(skm.accuracy_score(y_test[:, i], y_score_bin[:, i]))
    precision.append(skm.precision_score(y_test[:, i], y_score_bin[:, i]))
    f1.append(skm.f1_score(y_test[:, i], y_score_bin[:, i]))
    recall.append(skm.recall_score(y_test[:, i], y_score_bin[:, i]))
    cm = skm.confusion_matrix(y_test[:, i], y_score_bin[:, i])
    tn, fp, fn, tp = cm.ravel()
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)

accuracy = np.array(accuracy)
precision = np.array(precision)
f1 = np.array(f1)
recall = np.array(recall)
TP = np.array(TP)
TN = np.array(TN)
FP = np.array(FP)
FN = np.array(FN)
scores_df = pd.DataFrame({'class': np.arange(120),
                          'TP': TP,
                          'TN': TN,
                          'FP': FP,
                          'FN': FN,
                          'accuracy': accuracy,
                          'recall': recall,
                          'f1': f1,
                          'precision': precision})

scores_df.to_csv('scores_{}.csv'.format(model_type))
print(scores_df.mean())
best_classes = scores_df.nlargest(5, 'precision')['class'].values.tolist()

plt.figure(figsize=(12, 12))
for i in best_classes:
    plt.plot(fpr[i], tpr[i], label='class: {} AUC: {:.4}'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC dla top 5 klas pod względem dokładności predykcji')
plt.legend()
plt.savefig('roc_{}.png'.format(model_type))


for i in best_classes:
    print(getKeysByValue(valid_set.class_indices, i))

