from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.keras import  layers, models,optimizers
from keras.regularizers import l2

# Training Parameters
learning_rate = 0.0001
training_steps = 10000
batch_size = 64
display_step = 5

# Network Parameters
num_input = 64
timesteps = 64
input_shape = (num_input , timesteps , 1)
num_hidden = 64
num_classes = 2



# Create model
def cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 16), activation='relu',bias_initializer = "random_uniform" , kernel_initializer='random_uniform', input_shape=(64, 64, 1) , data_format = 'channels_last'))
    model.add(layers.MaxPooling2D((1,4)))
    model.add(layers.Conv2D(64, (2, 8),bias_initializer = "random_uniform" , kernel_initializer='random_uniform', activation='relu'))
    model.add(layers.MaxPooling2D((2, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,bias_initializer = "random_uniform" , kernel_initializer='random_uniform', activation='relu'))
    model.add(layers.Dense(2, bias_initializer = "random_uniform" ))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model

def crnn():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu',bias_initializer = "he_normal" , kernel_initializer='he_normal', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Reshape(( 64, 16*64)))
   # model.add(layers.GRU(128, return_sequences=True))
    model.add(layers.LSTM(64 ,go_backwards=True ,return_sequences=True) )
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu' , bias_initializer = "random_uniform" ))
    model.add(layers.Dense(1, activation='sigmoid', bias_initializer = "random_uniform" ))
    opt = optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model

def rnn():
    model = models.Sequential()

   # model.add(layers.GRU(64 , input_shape=(64,64) ,return_sequences=False , activation='relu',bias_initializer = "he_normal"))
   # model.add(layers.BatchNormalization())
 #   model.add(layers.Flatten(input_shape=(64,64)))
   # model.add(layers.Embedding(input_dim=64*64, output_dim=64))
  #  model.add(layers.GRU(32, return_sequences=True))
  #  model.add(layers.LSTM(16))
    #model.add(layers.GRU(128, bias_initializer = "random_uniform", return_sequences=False , input_shape=(64,64)))
    #model.add(layers.Flatten())
    #model.add(layers.Dense(128, activation='relu'))
    model.add(layers.SimpleRNN(128 , input_shape=(64,64) ,activation='relu', bias_initializer="random_uniform"))
    model.add(layers.Dense( 64, activation='relu' , bias_initializer = "random_uniform"))
    model.add(layers.Reshape((64,32)))
    model.add(layers.SimpleRNN(32 ,input_shape=(64), activation='relu'))
   # model.add(layers.SimpleRNN(64, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='glorot_uniform', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
    model.add(layers.Dense( 1, activation='linear' , bias_initializer = "random_uniform"))
   # model.add(layers.Dense(64, 2))
    opt = optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=opt,
              loss="binary_crossentropy",
              metrics=['accuracy'])
    #model.build();
    model.summary()
    return model

def SVM():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 16), activation='relu',bias_initializer = "random_uniform" , kernel_initializer='random_uniform', input_shape=(64, 64, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,kernel_regularizer=l2(0.01)))
    model.add(layers.Activation('linear'))
    model.compile(loss='hinge',
              optimizer="Adadelta",
              metrics=['accuracy'])
    model.summary()
    return model
