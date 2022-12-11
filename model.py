import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.utils.np_utils import to_categorical
import random,shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255),target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,color_mode='grayscale',class_mode=class_mode,target_size=target_size)


TS=(24,24)
train_batch= generator('data/train',shuffle=True,target_size=TS)
valid_batch= generator('data/valid',shuffle=True,target_size=TS)




model = Sequential([
#extration des caracteristiques
    #32 convolution filters  de taile 3*3 est etuliser et inpute et de taille (24*24*1)
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    #choisire la meilleur valeur avec pooling
    MaxPooling2D(pool_size=(1,1)),
    #again
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    #again cette fois avec 64 convolution
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),


    
    #allumer et éteindre les neurones au hasard pour améliorer la convergence
    Dropout(0.25),
    #aplatir
    Flatten(),
    #réseau de neurones (128 neurones)
    Dense(128, activation='relu'),
    #allumer et éteindre les neurones au hasard pour améliorer la convergence
    Dropout(0.5),
    #sortie d'un softmax pour écraser la matrice en probabilités de sortie
    Dense(2, activation='softmax')
])
 
#compiler le model cnn  -adam , -categorical_crossentropy(parce que la formation utilise la fonction de perte et l'optimiseur)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#lancer training avec 15 epochs
model.fit_generator(train_batch, validation_data=valid_batch,epochs=15)
#enregistrer le model (les poids optimales)
model.save('models/model.h5', overwrite=True)