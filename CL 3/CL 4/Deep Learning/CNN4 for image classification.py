#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Design and implement a CNN for Image Classification 


# In[1]:


# (a) Select a suitable image classification dataset

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)


# In[2]:


'''
(b) Optimize with hyperparameters
We’ll tweak:
Filter size
Number of layers
Optimizer (Adam vs SGD)
Dropout rate
Learning rate (through optimizer)
'''


# In[3]:


# Step 1: Build a CNN Model with Tunable Parameters

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),  # dropout added
    Dense(10, activation='softmax')  # 10 classes
])

# Try tweaking learning rate here
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[4]:


# Step 2: Train the Model

history = model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.1)


# In[5]:


# Step 3: Evaluate the Model

test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")


# In[6]:


# Step 4: Visualize Performance

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




