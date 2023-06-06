import pandas as pd

train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

train_df.head()

y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

x_train.shape
y_train.shape

x_valid.shape
y_valid.shape

import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    
    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')
   
   x_train.min()
   x_train.max()
  
import tensorflow.keras as keras
num_classes = 24
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
