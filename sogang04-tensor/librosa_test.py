import librosa.display
import numpy as np
import matplotlib.pyplot as plt

SOURCE = 'sogang04-tensor/rattle.wav.wav'
TARGET = ''

file = SOURCE
y, sr = librosa.load(file) # (default  sr=22050)
# Return
    # y: np.ndarray [shape=(n,) or (…, n)] / audio time series. Multi-channel is supported.
    # sr: number > 0 [scalar / sampling rate of
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# Returns
    # S: np.ndarray [shape=(…, n_mels, t)] / Mel spectrogram
S_dB = librosa.power_to_db(S, ref=np.max)
if TARGET == '':
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()
else:
    fig = plt.figure(figsize=(15, 15))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr)
    plt.savefig(TARGET, bbox_inches='tight', pad_inches=0)
    plt.close(fig) 
