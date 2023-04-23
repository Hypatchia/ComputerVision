import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define size of 
size = 30

# Define Generator
Generator = keras.models.Sequential(
  [
    keras.layers.Dense(100,activation = "selu",input_shape = [size]),
    keras.layers.Dense(150, activation = "selu"),
    keras.layers.Dense(28*28, activation ="sigmoid"),
    keras.layers.Reshape([28,28])
    ])

# Define Discriminator 

Discriminator = keras.models.Sequential(
  [
    keras.layers.Flatten(input_shape = [28,28]),
    keras.layers.Dense(150,activation = "selu"),
    keras.layers.Dense(100, activation = "selu"),
    keras.layers.Dense(1, activation ="sigmoid"),
  ])

# Concatenate Generator and Discriminator 
GAN = keras.models.Sequential([Generator,Discriminator])  

# Model Summary
GAN.summary()

# Compile Discriminator
Discriminator.compile(loss = "binary_crossentropy", optimizer = "rmsprop")


# Deactivate Discriminator 

Discriminator.trainable = False

# Compile GAN 

GAN.compile(loss = "binary_crossentropy", optimizer = "rmsprop")


# Load Data 
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
len(list(dataset.as_numpy_iterator()))

# Define Training loop
def Train_GAN(GAN ,dataset ,batch_size ,size ,n_epochs = 50):
  Generator , Discriminator = GAN.layers
  for epoch in range(n_epochs):
    for X_batch in dataset :
      print('batch i')
      #discriminator training :
      noise = tf.random.normal(shape = [batch_size , size])
      generated = Generator(noise)
      X_fake_nd_real = tf.concat([generated ,X_batch],axis = 0)
      y_fake_nd_real = tf.constant([[0.]]*batch_size+[[1.]]*batch_size)
      Discriminator.trainable = True
      Discriminator.train_on_batch(X_fake_nd_real,y_fake_nd_real)

      #generator training 
      noise = tf.random.normal(shape = [batch_size , size])
      y = tf.constant([[1.]]*batch_size)
      Discriminator.trainable = False
      GAN.train_on_batch(noise,y)




# Train Model
Train_GAN(GAN ,dataset ,batch_size ,size )