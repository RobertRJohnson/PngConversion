#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing import image as KI
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from PIL import Image
import os
import glob
import pathlib
import matplotlib.pyplot as plt


# In[2]:


print("Tensor Flow Version: {}".format(tf.__version__))
data_dir_train = "PATH OF TRAINING IMAGES"
data_dir_train = pathlib.Path(data_dir_train)
data_dir_val = "PATH TO TESTING IMAGES"
data_dir_val = pathlib.Path(data_dir_val)


# In[3]:


image_count_train = len(list(data_dir_train.glob('*/*')))

image_count_val = len(list(data_dir_val.glob('*/*')))

print(image_count_train)
print(image_count_val)


# In[4]:


CLASS_NAMES_TRAIN = np.array([item.name for item in data_dir_train.glob('*') if item.name != "LICENSE.txt"])


CLASS_NAMES_VAL = np.array([item.name for item in data_dir_val.glob('*') if item.name != "LICENSE.txt"])


print(CLASS_NAMES_TRAIN)

print(CLASS_NAMES_VAL)


# In[5]:


trdata = ImageDataGenerator(rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5)
traindata = trdata.flow_from_directory(directory="PATH OF TRAINING IMAGES",target_size=(224,224), shuffle = True, classes = list(CLASS_NAMES_TRAIN))
tsdata = ImageDataGenerator(rescale=1./255)
testdata = tsdata.flow_from_directory(directory="PATH TO TESTING IMAGES", target_size=(224,224), shuffle = True, classes = list(CLASS_NAMES_VAL))


# In[6]:


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES_TRAIN[label_batch[n]==1][0].title())
      plt.axis('off')
        
image_batch, label_batch = next(traindata)
show_batch(image_batch, label_batch)


# In[7]:


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES_VAL[label_batch[n]==1][0].title())
      plt.axis('off')
image_batch, label_batch = next(testdata)
show_batch(image_batch, label_batch)


# In[8]:


model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(.5))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(.5))


# In[9]:


model.add(Flatten())

model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=10, activation="softmax"))


# In[10]:


import tensorflow as tf
opt = Adam(lr=.00001)
model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


# In[11]:


model.summary()


# In[12]:


checkpoint = ModelCheckpoint("Drop_Reg.h5", monitor='val_accuracy', verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto')

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata,
                           validation_steps=100,epochs=100,callbacks=[checkpoint,early])


# In[13]:


class_dict = traindata.class_indices
print (class_dict)


def get_class(val): 
    for key, value in class_dict.items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist"


# In[14]:



plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


# In[15]:


img = tf.io.read_file("PATHTOQUERYIMG.jpg")
img = tf.image.decode_jpeg(img)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [224,224])
img = np.asarray(img)
print(img.shape)
plt.imshow(img)
img = np.expand_dims(img,0)
print(img.shape)
from tensorflow.keras.models import load_model
saved_model = load_model("Directory/Drop_Reg.h5")
output = saved_model.predict(img)
answer = get_class(np.argmax(output[0]))
print(np.argmax(output[0]))
plt.title(answer)
plt.show()


# In[ ]:





# In[18]:


img = tf.io.read_file("PATHTOQUERYIMG.jpg")
img = tf.image.decode_jpeg(img)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [224,224])
img = np.asarray(img)
print(img.shape)
plt.imshow(img)
img = np.expand_dims(img,0)
print(img.shape)
from tensorflow.keras.models import load_model
saved_model = load_model("Directory/Drop_Reg.h5")
output = saved_model.predict(img)
answer = get_class(np.argmax(output[0]))
print(np.argmax(output[0]))
plt.title(answer)
plt.show()


# In[ ]:


img = tf.io.read_file("PATHTOQUERYIMG.jpg")
img = tf.image.decode_jpeg(img)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [224,224])
img = np.asarray(img)
print(img.shape)
plt.imshow(img)
img = np.expand_dims(img,0)
print(img.shape)
from tensorflow.keras.models import load_model
saved_model = load_model("Directory/Drop_Reg.h5")
output = saved_model.predict(img)
answer = get_class(np.argmax(output[0]))
plt.title(answer)
plt.show()


# In[ ]:




