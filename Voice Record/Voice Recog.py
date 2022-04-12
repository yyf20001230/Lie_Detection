
import numpy as np
import soundfile
import sklearn
import librosa
import pickle
import glob
import os
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are

import warnings
warnings.filterwarnings('ignore')

def get_features(folder, mfcc = True, chroma = True, mel = True, contrast = True, tonnetz = True):

    with soundfile.SoundFile(folder) as sound_file:
        
        X = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(X)), sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(X)), sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# emolist = {
#     '01': 'neutral',
#     '02': 'calm',
#     '03': 'happy',
#     '04': 'sad',
#     '05': 'angry',
#     '06': 'fearful',
#     '07': 'disgust',
#     '08': 'surprised'
# }

# AVAIL_EMOTIONS = {
#     "angry",
#     "sad",
#     "neutral",
#     "happy",
#     "fearful",
#     "surprised"
# }

# def load_data():
#     x, y = [], []
#     a = 0
#     for i in glob.glob("data/Actor_*/*.wav"):
#         # get the base name of the audio file
#         name = os.path.basename(i)
#         # get the emotion label
#         emotion = emolist[name.split("-")[2]]
#         if emotion not in AVAIL_EMOTIONS:
#             continue
#         # extract speech features
#         features = get_features(i)
#         # add to data
#         x.append(features)
#         y.append(emotion)
#         print("Appended", a, "units")
#         a += 1
#     # split the data to training and testing and return it
#     return np.array(x), y

# a = load_data()

# pickle.dump(a[0], open('x.p', 'wb'))
# pickle.dump(a[1], open('y.p', 'wb'))

# x_train, x_test, y_train, y_test = train_test_split(pickle.load(open("x.p", "rb")), pickle.load(open("y.p", "rb")), test_size=0.3, random_state=7)


# print("[+] Number of training samples:", x_train.shape[0])
# print("[+] Number of testing samples:", x_test.shape[0])
# print("[+] Number of features:", x_train.shape[1])

# model_params = {
#     'alpha': 0.01,
#     'batch_size': 256,
#     'epsilon': 1e-08, 
#     'hidden_layer_sizes': (300,), 
#     'learning_rate': 'adaptive', 
#     'max_iter': 500, 
# }

# model = MLPClassifier(**model_params)

# print("[*] Training the model...")
# model.fit(x_train, y_train)

# pickle.dump(model, open('model.p', 'wb'))

# y_pred = model.predict(x_test)

# # calculate the accuracy
# accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# print("Accuracy: {:.2f}%".format(accuracy*100))

x = []
features = get_features("f.wav")
x.append(features)
res = np.array(x)

model = pickle.load(open("model.p", "rb"))

print(model.predict(x))

