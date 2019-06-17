import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import numpy as np
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

X_train = pd.read_csv('SVM_all_X_train.csv').values
y_train = pd.read_csv('SVM_all_y_train.csv').values

X_test = pd.read_csv('SVM_all_X_test.csv').values
y_test = pd.read_csv('SVM_all_y_test.csv').values

print(y_train.shape)
print(X_train.shape)

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import time
kernel = 'linear'
model_type = 'SVM_all_' + kernel
svm_model_linear = SVC(kernel=kernel, C=1, probability=True, degree=2)
clf = OneVsRestClassifier(svm_model_linear)

start = time.time()
clf.fit(X_train, y_train)
print('time: {} s'.format(time.time() - start))

y_score = clf.predict_proba(X_test)

print(y_test.shape)
print(y_score.shape)
print(y_score[0])
print()

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


from keras.preprocessing.image import ImageDataGenerator
valid_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_set = valid_datagen.flow_from_directory('dataset/valid_set',
                                              target_size=(img_width, img_width),
                                              batch_size=batch_size,
                                              class_mode='categorical')
for i in best_classes:
    print(getKeysByValue(valid_set.class_indices, i))
