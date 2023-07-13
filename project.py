#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import ipywidgets as widgets
import io
from PIL import Image
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import os
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import roc_curve, auc
import ipywidgets as widgets
import io
from PIL import Image
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import os


# In[2]:


import os
path = os.listdir('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}
import cv2
X = []
Y = []
for cls in classes:
    pth = 'C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[3]:


X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)
np.unique(Y)
pd.Series(Y).value_counts()


# In[4]:


X.shape, X_updated.shape


# In[5]:


plt.imshow(X[0], cmap='gray')


# In[6]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[7]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)


# In[8]:


xtrain.shape, xtest.shape


# In[9]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[10]:


from sklearn.decomposition import PCA


# In[11]:


print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[13]:


import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)


# In[14]:


ypred = lg.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
report = classification_report(ytest, ypred)
print('Accuracy:', accuracy)
print("Logistic Regression Classification report:\n", report)


# In[15]:


labels = ['No Tumour', 'Pituitary Tumor']
cm = confusion_matrix(ytest, ypred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='RdGy', xticklabels=labels, yticklabels=labels, ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[16]:


sv = SVC()
sv.fit(xtrain, ytrain)


# In[17]:


print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))


# In[18]:


print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))


# In[19]:


pred = sv.predict(xtest)


# In[20]:


misclassified=np.where(ytest!=pred)
misclassified


# In[21]:


print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])


# In[22]:


dec = {0:'No Tumor', 1:'Positive Tumor'}


# In[23]:


svm_classifier = SVC(kernel='linear', random_state=101)
svm_classifier.fit(xtrain, ytrain)
svm_y_pred = svm_classifier.predict(xtest)
svm_accuracy = accuracy_score(ytest, pred)
svm_confusion = confusion_matrix(ytest, pred)
svm_report = classification_report(ytest, pred)


# In[24]:


print("SVM Accuracy:", svm_accuracy)
print("SVM Classification report:\n", svm_report)


# In[25]:


labels = ['No Tumour', 'Pituitary Tumor']
cm = confusion_matrix(ytest, pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[26]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/training/')
c=1
for i in os.listdir('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/training/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/training/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[27]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/testing/')
c=1
for i in os.listdir('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/braindata/testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[28]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score


# In[29]:



X_train = []
Y_train = []
image_size = 200
labels = ['no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/brain_tumor_dataset/data/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('C:/Users/smrit/OneDrive/Desktop/rese/braincancer/brain_tumor_dataset/data/Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[30]:


X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape


# In[31]:


X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=101)


# In[32]:


y_train_new = []
#ml should have binary values
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


# In[33]:


model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(200,200,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))


# In[34]:


model.summary()


# In[35]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


history=model.fit(X_train,y_train,epochs=10,validation_split=0.25)


# In[ ]:


cnn_acc=accuracy_score(y_test,np.round(model.predict(X_test)))


# In[ ]:


print("Accuracy of CNN",cnn_acc)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
labels = ['CNN', 'SVM']
accuracy_scores = [cnn_acc, svm_accuracy]
plt.bar(labels, accuracy_scores)
plt.title('Comparison of CNN and SVM on Brain Dataset')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


'''labels = ['No Tumour', 'Pituitary Tumor']
cr = confusion_matrix(y_test,np.round(model.predict(X_test)))
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cr, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()'''
labels = ['No Tumour', 'Pituitary Tumor']
y_pred = np.argmax(model.predict(X_test), axis=-1)
y_test_single = np.argmax(y_test, axis=-1)
cm = confusion_matrix(y_test_single, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='viridis', xticklabels=labels, yticklabels=labels, ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


report = classification_report(y_test_single, y_pred)
print(report)


# In[ ]:


labels = ['Logistic Regression', 'SVM', 'CNN']
accuracy = [accuracy, svm_accuracy, cnn_acc]
plt.bar(labels, accuracy)
plt.title('Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:



# Compute FPR and TPR for each model
svm_fpr, svm_tpr, svm_thresholds = roc_curve(ytest, pred)
roc_auc_svm = auc(svm_fpr, svm_tpr)
cnn_fpr, cnn_tpr, cnn_thresholds = roc_curve(y_test_single, y_pred)
roc_auc_cnn = auc(cnn_fpr, cnn_tpr)
lr_fpr, lr_tpr, lr_thresholds = roc_curve(ytest, ypred)
roc_auc_lr = auc(lr_fpr, lr_tpr)

plt.plot(svm_fpr, svm_tpr, color='blue', lw=2, label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot(cnn_fpr, cnn_tpr, color='green', lw=2, label='CNN (AUC = %0.2f)' % roc_auc_cnn)
plt.plot(lr_fpr, lr_tpr, color='red', lw=2, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




