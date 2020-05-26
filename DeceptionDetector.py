#!/usr/bin/env python

import sys
from DDetector import *
from sklearn.model_selection import train_test_split
sys.path.append('utils/')
from itertools import zip_longest
import fsdd
import MyCustomCallback
import matplotlib.pyplot as plt
import numpy as np


DeceptionDB_path = "data/DeceptionDB/"
DeceptionDB_csv_path = "data/DeceptionDB/description.csv"

test_db_data_path = "data/num_rec_data/"
test_db_spectro_data_path = test_db_data_path + 'spectro/'

#contains spectrograms from wav files. same file name different ending wav <--> png
spectrogram_dir = "data/spectrograms/"
batch_size = 64
data_chunks = 1

def showplt(history):
    print(history.history.keys())
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def main():
    print("running DDetector")
    db = fsdd.FSDD(spectrogram_dir)
    audio_list , label_list = db.get_spectrograms(spectrogram_dir)

    data_audio_train, data_audio_test , data_label_train , data_label_test = train_test_split(audio_list , label_list , test_size=0.20 , shuffle = True)

    model = cnn_example.crnn()
    history = model.fit(data_audio_train, data_label_train, epochs=2,
                    validation_data=(data_audio_test, data_label_test) , callbacks = [MyCustomCallback.MyCustomCallback(data_audio_test, data_label_test)])
    model.save('crnn_bad_preff')
    showplt(history)

  #  pred = model.predict_classes(data_audio_test, verbose=2)
  #  printPref(pred,data_label_test)





if __name__ == "__main__":
    main()

