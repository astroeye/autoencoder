'''
1. Extract spectrograms from wav files
2. Load training images
3. Build autoencoder
4. Set threshold
5. Make an inference
6. [Web Interface] Copy wav files from source to target data
7. [Web Interface] Select a day and make inference
'''

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


def create_training_data(data_path, size=224):
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
            print("an error has occured: ", err, str(full_name))

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
            layers.Reshape((224, 224))
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


def spectrogram_loss(autoencoder, spectrogram, size=224):
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
    1. Extract spectrograms from wav files
    '''
    SOURCE = 'D:/Data/36cc'
    TARGET = 'D:/Data/36cc_converted'
    FIG_SIZE = (20, 20)

    extractor = SpectrogramExtractor()
    extractor.extract(SOURCE, TARGET, FIG_SIZE)

    '''
    2. Load training images
    '''
    data_path = "D:/Data/out"
    x_train = create_training_data(data_path)

    data_path = "D:/Data/out/test"
    x_test = create_training_data(data_path)

    '''
    3. Build autoencoder
    '''
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

k = 4
num_val_samples = len(x_train) // k
num_epochs = 100
all_scores = []

 autoencoder = Autoencoder(latent_dim=64 * 3)
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

   for i in range(k):
        print(f"Processing fold #{i}")
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        # val_targets = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        history = autoencoder.fit(partial_train_data, partial_train_data,
                                  epochs=num_epochs,
                                  shuffle=True,
                                  #   validation_data=(val_data, val_data))
                                  validation_data=(val_data, val_data))

        # Evaluate the model
        all_scores.append(history.history["val_loss"])

        # a summary of architecture
        autoencoder.encoder.summary()
        autoencoder.decoder.summary()

        # plot history
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.show()

    '''
    Evaluate all_scores
    '''
    all_scores
    np.mean(all_scores)

    '''
    Save a complete model
    '''
    # # save and load a mode
    # autoencoder.save('./model/')
    # autoencoder = keras.models.load_model('./model/')

    # calculate loss and threshold
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    '''
    4. Set threshold
    '''
    loss = tf.keras.losses.mse(decoded_imgs, x_train)
    threshold = np.mean(loss) + np.std(loss)
    print("mean: ", round(np.mean(loss), 4))
    print("standard deviation: ", round(np.std(loss), 4))
    print("Loss Threshold: ", round(threshold, 4))

    # load autoencoder model
    if autoencoder is None:
        autoencoder = keras.models.load_model('./model/')

    threshold = np.mean(loss) + np.std(loss)

    '''
    5. Make an inference
    '''
    file = 'd:/data/out/36cc/36cc_OK_2208211119H0010019498.jpg'   # good
    file = 'd:/data/out/36cc/test/36cc_OK_121282111NB982000290918.jpg'   # anomaly

    # get statistics for each spectrogram
    sample = plt.imread(file)
    plt.imshow(sample)
    sample = pathlib.Path(file)
    sample_loss = spectrogram_loss(autoencoder, sample)

    if sample_loss > threshold:
        print(
            f'Loss is greater than threshold \n \
              Sample Loss: {sample_loss} \n \
              Threshold: {threshold} ')
    else:
        print(
            f'Loss is lesser than threshold \n \
              Sample Loss: {sample_loss} \n \
              Threshold: {threshold} ')
