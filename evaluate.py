import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
import utils.spectogramer as spectrogramer

model_path: str = "C:\\Users\\neite\\OneDrive\\Documents\\schooling\\cheat_detector_model\\crnn_bad_preff"


def main():
    claim_path = sys.argv[1]
    spectrogramer.wav_to_spectrogram(claim_path,claim_path.replace('.wav', '.png') )

    spectrogram = imageio.imread(claim_path.replace('.wav', '.png'), pilmode='F')
    spectrogram = np.reshape(spectrogram, (1,64, 64, 1))
    loaded_model = tf.keras.models.load_model(model_path, compile=True, custom_objects=None)
    pred = loaded_model.predict_classes(spectrogram)

    print(pred)


if __name__ == "__main__":
    main()
