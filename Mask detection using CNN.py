#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow


# In[2]:


import os
from keras.preprocessing import image
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras import Sequential


# ## Preprocessing ##

# In[3]:


categories = ['with_mask','without_mask']


# In[4]:


data = []
for category in categories:
    path = os.path.join('train',category)
    
    label = categories.index(category)
    
    for file in os.listdir(path):
        img_path = os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        
        data.append([img,label])


# In[5]:


random.shuffle(data)


# In[6]:


x = []
y = []

for features,label in data:
    x.append(features)
    y.append(label)


# In[7]:


x = np.array(x)
y = np.array(y)


# In[8]:


x = x/255


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[10]:


vgg = VGG16()


# In[9]:


model = Sequential()


# In[12]:


vgg.summary()


# In[13]:


for layer in vgg.layers[:-1]:
    model.add(layer)


# In[14]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[15]:


model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))


# In[21]:


cap = cv2.VideoCapture(0)


# In[22]:


def detect_face_mask(img):
    ypred = model.predict(img.reshape(1,224,224,3))
    return ypred


# In[23]:


def draw_label (img,text,pos,bg_color):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x = pos[0]+ text_size[0][0] + 2
    end_y = pos[1]+ text_size[0][1] - 2
    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)


# In[ ]:


while True:
    
    ret,frame = cap.read()
    
    img = cv2.resize(frame,(224,224))
    
    ypred= detect_face_mask(img)
    
    if ypred.all() == 0:
       
        
        draw_label(frame,"Mask",(30,30),(0,255,0))
    else:
        
        draw_label(frame,"No Mask",(30,30),(0,0,255))
        
    cv2.imshow('window',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
cv2.destroyAllWindows()


# In[ ]:




