"""Installing missing modules if needed"""
# pip install keras
# pip install pillow
# pip install tensorflow
# pip install sklearn
# pip install pandas
# pip install h5py

"""Imports"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

"""Uncomment to train, Comment to test"""

# """Load data and reshape dims"""
#
# IMG_DIM = (150, 150)
#
# if not os.path.exists('dataset'):
#     print("Dataset unavailable: download from https://www.kaggle.com/c/dogs-vs-cats/data and move to same directory")
# else:
#     train_files = glob.glob('dataset/trainingData/*')
#
#     train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
#     train_imgs = np.array(train_imgs)
#     train_labels = [os.path.basename(fn).split('.')[0].strip() for fn in train_files]
#
#     validation_files = glob.glob('dataset/validationData/*')
#     validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
#     validation_imgs = np.array(validation_imgs)
#     validation_labels = [os.path.basename(fn).split('.')[0].strip() for fn in validation_files]
#
#     print('Train dataset shape:', train_imgs.shape,
#           '\tValidation dataset shape:', validation_imgs.shape)
#
#
# """Rescale 8-bit pixels between (0,1)"""
#
# train_imgs_scaled = train_imgs.astype('float32')
# validation_imgs_scaled = validation_imgs.astype('float32')
# train_imgs_scaled /= 255
# validation_imgs_scaled /= 255
#
# print(train_imgs[0].shape)
# array_to_img(train_imgs[0])
#
#
# """"Encode text class labels to numeric"""
#
# from sklearn.preprocessing import LabelEncoder
#
# le = LabelEncoder()
# le.fit(train_labels)
# train_labels_enc = le.transform(train_labels)
# validation_labels_enc = le.transform(validation_labels)
#
# print(train_labels[1495:1505], train_labels_enc[1495:1505])
#
#
# """Set hyperparams"""
#
# batch_size = 32
# num_classes = 2
# epochs = 8
# input_shape = (150, 150, 3)
#
#
# """(1) Basic CNN model"""
#
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
# from keras.models import Sequential
# from keras import optimizers
#
# model = Sequential()
#
# model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(),
#               metrics=['accuracy'])
#
# model.summary()
#
#
# """Train and validate"""
#
# history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
#                     validation_data=(validation_imgs_scaled, validation_labels_enc),
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1)
#
#
# """Plot model accuracy and errors"""
#
#
# def plotLearningHistory(history, title="Learning History"):
#     epochs = len(history.history['accuracy'])
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#     t = f.suptitle(title, fontsize=12)
#     f.subplots_adjust(top=0.85, wspace=0.3)
#
#     epoch_list = list(range(1, epochs+1))
#     ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
#     ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
#     ax1.set_xticks(np.arange(0, epochs+1, 5))
#     ax1.set_ylabel('Accuracy Value')
#     ax1.set_xlabel('Epoch')
#     ax1.set_title('Accuracy')
#     l1 = ax1.legend(loc="best")
#
#     ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
#     ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
#     ax2.set_xticks(np.arange(0, epochs+1, 5))
#     ax2.set_ylabel('Loss Value')
#     ax2.set_xlabel('Epoch')
#     ax2.set_title('Loss')
#     l2 = ax2.legend(loc="best")
#
#
# """Plot learning history"""
#
# plotLearningHistory(history, title='Basic CNN Performance')
#
#
# """(2) CNN Model with regularization
# Using Dropout.
# Dropout is a technique where randomly selected neurons are ignored during training.
# You can imagine that if neurons are randomly dropped out of the network during training, that other neurons will have
# to step in and handle the representation required to make predictions for the missing neurons. This in turn results in
# a network that is capable of better generalization and is less likely to overfit the training data.
# """
#
#
# model = Sequential()
#
# model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
#
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(),
#               metrics=['accuracy'])
#
# model.summary()
#
#
# history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
#                     validation_data=(validation_imgs_scaled, validation_labels_enc),
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1)
#
# plotLearningHistory(history, title='CNN with Dropout Regularization')
#
#
# """Save the model"""
#
# model.save('cats_dogs_basic_cnn.h5')
#
#
# """(3) CNN Model with Image Augmentation"""
# """ Image augmentation with Keras ImageDataGenerator.
# Only on train data"""
#
# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')
#
# val_datagen = ImageDataGenerator(rescale=1./255)
#
#
# """Visualise ImageDataGenerator"""
#
# img_id = 50
# cat_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
#                                    batch_size=1)
# cat = [next(cat_generator) for i in range(0,5)]
#
# fig, ax = plt.subplots(1,5, figsize=(16, 6))
# print('Labels:', [item[1][0] for item in cat])
# l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
#
#
# """Use same model architecture with the new data generators"""
#
# train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
# input_shape = (150, 150, 3)
#
# model = Sequential()
#
# model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['accuracy'])
#
# history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=20,
#                               validation_data=val_generator, validation_steps=50,
#                               verbose=1)
#
# """Plot performance and save the model"""
#
# plotLearningHistory(history, title='CNN with Image Augmentation Performance')
# model.save('cats_dogs_cnn_img_aug.h5')
#
#
# """Transfer Learning with pre-trained CNN Models
# VGG-16
# As (1) a feature extractor (all Conv layers frozen) and as (2) a fine-tuned model (last 2 blocks open to learning)"""
#
# """(4) Pre-trained CNN Model as a Feature Extractor
# Load the model and freeze the convolution blocks"""
#
# from keras.applications import vgg16
# from keras.models import Model
# import keras
#
# vgg = vgg16.VGG16(include_top=False, weights='imagenet',
#                   input_shape=input_shape)
#
# output = vgg.layers[-1].output
# output = keras.layers.Flatten()(output)
# vgg_model = Model(vgg.input, output)
#
# vgg_model.trainable = False
# for layer in vgg_model.layers:
#     layer.trainable = False
#
# import pandas as pd
#
# pd.set_option('max_colwidth', -1)
# layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
# pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
#
#
# """Visualize one bottleneck feature (out of 512) for a sample image"""
#
# bottleneck_feature_example = vgg.predict(train_imgs_scaled[0:1])
# print(bottleneck_feature_example.shape)
# plt.imshow(bottleneck_feature_example[0][:,:,0])
#
#
# """Extract out the bottleneck features (flattened) from our training and validation sets
# CAREFUL: long computation time (around 15min)"""
#
#
# def get_bottleneck_features(model, input_imgs):
#     features = model.predict(input_imgs, verbose=0)
#     return features
#
#
# train_features_vgg = get_bottleneck_features(vgg_model, train_imgs_scaled)
# validation_features_vgg = get_bottleneck_features(vgg_model, validation_imgs_scaled)
#
# print('Train Bottleneck Features:', train_features_vgg.shape,
#       '\tValidation Bottleneck Features:', validation_features_vgg.shape)
#
#
# """Build the CNN architecture that will take these features as input
# Which means we only build the Dense FC layers to regress these features (8192)."""
#
# input_shape = vgg_model.output_shape[1]
#
# model = Sequential()
# model.add(InputLayer(input_shape=(input_shape,)))
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['accuracy'])
#
# model.summary()
#
#
# """Train the model (FC layers)"""
#
# history = model.fit(x=train_features_vgg, y=train_labels_enc,
#                     validation_data=(validation_features_vgg, validation_labels_enc),
#                     batch_size=batch_size,
#                     epochs=30,
#                     verbose=1)
#
#
# """Plot performance and save the model"""
#
# plotLearningHistory(history, title='Pre-trained CNN')
# model.save('cats_dogs_tlearn_basic_cnn.h5')
#
#
# """(5) Pre-trained CNN Model as a Feature Extractor with Image Augmentation"""
#
# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')
#
# val_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
#
#
# """Pass the VGG Model instead of extracting the bottleneck features since we are now working with generators (random
# and not constant)"""
#
# model = Sequential()
# model.add(vgg_model)
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=2e-5),
#               metrics=['accuracy'])
#
# history_4 = model.fit_generator(train_generator, steps_per_epoch=100, epochs=12,
#                                 validation_data=val_generator, validation_steps=50,
#                                 verbose=1)
#
# model.save('cats_dogs_tlearn_img_aug_cnn_.h5')
# plotLearningHistory(history, title='Pre-trained CNN with Image augmentation')
#
#
# """(6) Pre-trained CNN Model with Fine-tuning and Image Augmentation
# Unfreeze convolutional block 4 and 5"""
#
# vgg_model.trainable = True
#
# set_trainable = False
# for layer in vgg_model.layers:
#     if layer.name in ['block5_conv1', 'block4_conv1']:
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
# layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
# pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
#
#
# """Using same model architecture and data generators as last model"""
#
# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')
# val_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
#
# model = Sequential()
# model.add(vgg_model)
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-5),
#               metrics=['accuracy'])
#
# history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=20,
#                               validation_data=val_generator, validation_steps=50,
#                               verbose=1)
#
# model.save('cats_dogs_tlearn_finetune_img_aug_cnn.h5')
# plotLearningHistory(history, title='Pre-trained CNN with fine-tuning')


"""
Evaluating the Models on Test Data
The Models were not trained properly because of a lack of computation power. But the idea is here.

Using model_evaluation_utils from Dipanjan Sarkar
"""

from keras.models import load_model
import model_evaluation_utils as meu

# load saved models
basic_cnn = load_model('cats_dogs_basic_cnn.h5')
img_aug_cnn = load_model('cats_dogs_cnn_img_aug.h5')
tl_cnn = load_model('cats_dogs_tlearn_basic_cnn.h5')
tl_img_aug_cnn = load_model('cats_dogs_tlearn_img_aug_cnn_.h5')
tl_img_aug_finetune_cnn = load_model('cats_dogs_tlearn_finetune_img_aug_cnn.h5')

# load other configurations
IMG_DIM = (150, 150)
input_shape = (150, 150, 3)
num2class_label_transformer = lambda l: ['cat' if x == 0 else 'dog' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'cat' else 1 for x in l]

# load VGG model for bottleneck features
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                  input_shape=input_shape)
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)
vgg_model.trainable = False


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features


"""Prepare test data"""

IMG_DIM = (150, 150)

test_files = glob.glob('dataset/testData/*')
#test_files = glob.glob('meme/testData/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
test_labels_enc = class2num_label_transformer(test_labels)

print('Test dataset shape:', test_imgs.shape)
print(test_labels[0:5], test_labels_enc[0:5])


"""Model 1: Basic CNN Performance"""

predictions = basic_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))


"""Model 2: Basic CNN with Image Augmentation Performance"""

predictions = img_aug_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))


"""Model 3:  Pre-trained CNN as a Feature Extractor Performance"""

test_bottleneck_features = get_bottleneck_features(vgg_model, test_imgs_scaled)
predictions = tl_cnn.predict_classes(test_bottleneck_features, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))


"""Model 4: Pre-trained CNN as a Feature Extractor with Image Augmentation Performance"""

predictions = tl_img_aug_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))


"""Model 5: Pre-trained CNN with Fine-tuning and Image Augmentation Performance"""

predictions = tl_img_aug_finetune_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))


"""Plot ROC Curves of worst and best models"""

# worst model - basic CNN
meu.plot_model_roc_curve(basic_cnn, test_imgs_scaled,
                         true_labels=test_labels_enc,
                         class_names=[0, 1])

# best model - transfer learning with fine-tuning & image augmentation
meu.plot_model_roc_curve(tl_img_aug_finetune_cnn, test_imgs_scaled,
                         true_labels=test_labels_enc,
                         class_names=[0, 1])
