#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Load Dataset from scikit-learn library


# In[4]:


from sklearn.datasets import load_iris


# In[5]:


datasets = load_iris()


# In[10]:


# Input Numbers


# In[11]:


data   = datasets.data
target = datasets.target


# In[16]:


import numpy as np
np.unique(target)


# In[13]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# In[20]:


# Create a Model
model = Sequential()
# 1st Hidden Layer
model.add(Dense(8, input_dim = 4, activation = 'relu'))
# 2nd Hidden Layer
model.add(Dense(8, input_dim = 8, activation = 'relu'))
# Final Layer
model.add(Dense(3, input_dim = 8, activation = 'softmax'))


# In[60]:


model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ['accuracy'])


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


train_data, test_data, train_target, test_target = train_test_split(data,target, test_size = 0.2)


# In[63]:


from tensorflow.keras.utils import to_categorical


# In[64]:


update_train_target = to_categorical(train_target)


# In[73]:


print(update_train_target[:5])


# In[66]:


history = model.fit(train_data, update_train_target, epochs = 5)


# In[67]:


import matplotlib.pyplot as plt


# In[68]:


plt.plot(history.history['loss'])
plt.xlabel("Iterations")
plt.ylabel("Loss Values")
plt.show()


# In[69]:


plt.plot(history.history['accuracy'])
plt.xlabel("Iterations")
plt.ylabel("Accuracy Values")
plt.show()


# In[74]:


predicted_target = model.predict(test_data)


# In[75]:


print(predicted_target)


# In[82]:


print("Pre-Labels:    ",np.argmax(predicted_target,axis=1))
print("Actual Labels: ",test_target)

