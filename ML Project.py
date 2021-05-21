#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[4]:


print(x_train[17])


# In[5]:


import matplotlib.pyplot as plt


# In[19]:


plt.imshow(x_train[15],cmap=plt.cm.binary)
plt.show()


# In[16]:


print(y_train[15])


# In[20]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train[9]


# In[21]:


print(x_train[10])
plt.imshow(x_train[10],cmap=plt.cm.binary)
plt.show()


# In[25]:


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# In[26]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[27]:


model.fit(x_train, y_train, epochs=9)


# In[28]:


model.save('Model')


# In[29]:


New_Model = tf.keras.models.load_model('Model')


# In[30]:


predictions=New_Model.predict(x_test)
print(predictions)


# In[37]:


import numpy as np
print(np.argmax(predictions[7]))


# In[36]:


plt.imshow(x_test[3],cmap=plt.cm.binary)
plt.show()


# In[ ]:




