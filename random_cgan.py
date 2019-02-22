# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:53:45 2019

@author: 
"""

import os
from __future__ import print_function, division
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
from skimage import data_dir,io,transform,color
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('th')


basepath = 'F:/data/Random/20190111/'
inputpath = basepath+'Inputs/'
outputpath = basepath+'Outputs/'
imagesize = 100

savepath = basepath+'save/cgan/'
if os.path.exists(savepath) == False:
    os.makedirs(savepath)
image_path = savepath+'images/'
if os.path.exists(image_path) == False:
    os.makedirs(image_path)
model_path = savepath+'models/'
if os.path.exists(model_path) == False:
    os.makedirs(model_path)


# Load the dataset
x = np.load(basepath+'save/dcgan/x.p.npy')
y = np.load(basepath+'save/dcgan/y.p.npy')

# Configure input
x = (x.astype(np.float32) - 127.5) / 127.5
x = np.expand_dims(x, axis=3)
#y_train = y_train.reshape(-1, self.dmd_size)

X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.1, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Deterministic output.
np.random.seed(1000)

# Input shape
img_rows = 100
img_cols = 100
channels = 1
img_shape = (img_rows, img_cols, channels)
dmd_size = 20*20
g_embedding_dim = 10
d_embedding_dim = 25
latent_dim = dmd_size*g_embedding_dim

optimizer = Adam(0.0002, 0.5)


def build_generator():

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(dmd_size,), dtype='int32')
    label_embedding = Flatten()(Embedding(2, g_embedding_dim, input_length = dmd_size)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)


def build_discriminator():

    model = Sequential()

    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(dmd_size,), dtype='int32')

    label_embedding = Flatten()(Embedding(2, d_embedding_dim, input_length = dmd_size)(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])
    #model_input = tf.reshape(model_input, (img_shape)) #

    validity = model(model_input)

    return Model([img, label], validity)



# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss=['binary_crossentropy'],
    optimizer=optimizer,
    metrics=['accuracy'])

# Build the generator
generator = build_generator()

noise = Input(shape=(latent_dim,))
label = Input(shape=(dmd_size,))
img = generator([noise, label])

discriminator.trainable = False

valid = discriminator([img, label])

combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'],
    optimizer=optimizer)


dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_path+'dcgan_loss_epoch_%d.png' % epoch)


def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    sampled_labels = np.random.randint(0,2,(examples, dmd_size))
    generatedImages = generator.predict([noise, sampled_labels])

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_path+'dcgan_generated_image_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save(model_path+'dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save(model_path+'dcgan_discriminator_epoch_%d.h5' % epoch)


def train(epochs, batch_size=128, sample_interval=50):

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a half batch of new images
        gen_imgs = generator.predict([noise, labels])

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Condition on labels 
        noise = np.random.normal(0, 1, (batch_size, 100))
        sampled_labels = np.random.randint(0,2,(batch_size, dmd_size))

        # Train the generator
        g_loss = combined.train_on_batch([noise, sampled_labels], valid)


        # Store loss of most recent batch from this epoch
        dLosses.append(d_loss)
        gLosses.append(g_loss)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            plotGeneratedImages(epoch)
            saveModels(epoch)
    # Plot losses from every epoch
    plotLoss(epoch)



if __name__ == '__main__':
    train(epochs=20000, batch_size=32, sample_interval=200)

