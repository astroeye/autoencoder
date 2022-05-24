from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
import librosa
import librosa.display
# from PIL import Image, ImageOps
from sklearn.model_selection import cross_val_score


'''
Read wav files from SOURCE folder, extract spectrograms in JPG format, and save in TARGET folder
'''

'''
1. Extract spectrograms from wav files
'''


class SpectrogramExtractor:
    def extract(self, SOURCE, TARGET, FIG_SIZE):
        os.chdir(SOURCE)
        for file in os.listdir(SOURCE):
            # load audio file with Librosa
            signal, sample_rate = librosa.load(file, sr=22050)

            # perform Fourier transform (FFT -> power spectrum)
            fft = np.fft.fft(signal)

            # calculate abs values on complex numbers to get magnitude
            spectrum = np.abs(fft)

            # create frequency variable
            f = np.linspace(0, sample_rate, len(spectrum))

            # take half of the spectrum and frequency
            left_spectrum = spectrum[:int(len(spectrum)/2)]
            left_f = f[:int(len(spectrum)/2)]

            # STFT -> spectrogram
            hop_length = 512  # in num. of samples
            n_fft = 2048  # window in num. of samples

            # calculate duration hop length and window in seconds
            hop_length_duration = float(hop_length)/sample_rate
            n_fft_duration = float(n_fft)/sample_rate

            # perform stft
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

            # calculate abs values on complex numbers to get magnitude
            spectrogram = np.abs(stft)  # np.abs(stft) ** 2

            # apply logarithm to cast amplitude to Decibels
            log_spectrogram = librosa.amplitude_to_db(spectrogram)

            # Matplotlib plots: removing axis, legends and white spaces
            plt.figure(figsize=FIG_SIZE)
            plt.axis('off')
            librosa.display.specshow(
                log_spectrogram, sr=sample_rate, hop_length=hop_length)

            data_path = pathlib.Path(TARGET)
            file_name = f'{file[0:-4]}.jpg'
            full_name = str(pathlib.Path.joinpath(data_path, file_name))
            plt.savefig(str(full_name), bbox_inches='tight', pad_inches=0)


'''
2. Load training images  
'''
# resize and normalize data for training


def create_training_data(data_path, size=224*8):
    training_data = []
    # for category in CATEGORIES:  # "baseline" and "rattle"

    #     path = os.path.join(data_path, category)  # create path
    #     # get the classification  (0 or a 1). 0=baseline 1=rattle
    #     class_index = CATEGORIES.index(category)

    # iterate over each image
    for image in os.listdir(data_path):
        try:
            data_path = pathlib.Path(data_path)
            full_name = str(pathlib.Path.joinpath(data_path, image))
            data = cv2.imread(str(full_name), 0)
            # resize to make sure data consistency
            resized_data = cv2.resize(data, (size, size))
            # add this to our training_data
            training_data.append([resized_data])
        except Exception as err:
            print("an error has occureC: ", err, str(full_name))

    # normalize data
    training_data = np.array(training_data)/255.
    # reshape
    training_data = np.array(training_data).reshape(-1, size, size)
    return training_data


'''
3. Build autoencoder 
'''
# Define a convolutional Autoencoder


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # input layer
        self.latent_dim = latent_dim
        # 1st dense layer
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(224*224, activation='sigmoid'),
            layers.Reshape((224*8, 224*8))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


'''
4. Set threshold
'''


def threshold(autoencoder, x_train):
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    loss = tf.keras.losses.mse(decoded_imgs, x_train)
    threshold = np.mean(loss) + np.std(loss)
    return threshold


'''
5. Make an inference
'''


def spectrogram_loss(autoencoder, spectrogram, size=224*8):
    data = np.ndarray(shape=(1, size, size), dtype=np.float32)
    # individual sample
    # Load an image from a file
    data = cv2.imread(str(spectrogram), 0)
    # resize to make sure data consistency
    resized_data = cv2.resize(data, (size, size))
    # nomalize img
    normalized_data = resized_data.astype('float32') / 255.
    # test an image
    encoded = autoencoder.encoder(normalized_data.reshape(-1, size, size))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    return sample_loss


if __name__ == "__main__":

    '''
    2. Load training images
    '''
    # ok
    data_path = "C:/Data/36cc_converted"
    X_train_ok = create_training_data(data_path)
    y_train_ok = np.zeros(int(len(X_train_ok))).astype(int).reshape((-1,1))

    # not_ok
    data_path = "C:/Data/36cc_converted_not_ok"
    X_train_not_ok = create_training_data(data_path)
    y_train_not_ok = np.ones(int(len(X_train_not_ok))).astype(int).reshape((-1,1))

    # concatenate ok and not_ok
    X = np.concatenate((X_train_ok, X_train_not_ok))
    y = np.concatenate((y_train_ok, y_train_not_ok))

    # test
    data_path = "C:/Data/36cc_converted_test"
    X_train_test = create_training_data(data_path)
    y_train_test = np.zeros(int(len(X_train_test))).astype(int).reshape((-1,1))

    '''
    3. Build model 
    '''
    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return model

    # build a simple model with Dense layer
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(X, y, epochs=4, batch_size=512)
    results = model.evaluate(X_train_test, y_train_test)

    '''
    K-fold validation
    '''
    k = 4
    num_val_samples = len(X) // k
    num_epochs = 100
    all_scores = []

    for i in range(k):
        print(f"Processing fold #{i}")
        val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [X[:i * num_val_samples],
             X[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y[:i * num_val_samples],
             y[(i + 1) * num_val_samples:]],
            axis=0)
        model.fit(partial_train_data, partial_train_targets,
                  epochs=num_epochs,
                  batch_size=16, verbose=0)
        val_mse, val_mae = model.evaluate(
            val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

        # Evaluate all_scores
        all_scores
        np.mean(all_scores)

    '''
    Saving the validation logs at each fold
    '''
    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print(f"Processing fold #{i}")
        val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [X[:i * num_val_samples],
                X[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y[:i * num_val_samples],
                y[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=16, verbose=0)
        mae_history = history.history["val_mae"]
        all_mae_histories.append(mae_history)

    # Building the history of successive mean K-fold validation scores
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    # Plotting validation scores
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
