import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import librosa
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")


def extract_features(audio):

    signal, sr = librosa.load(audio)
    mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr = sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    comprehensive_mfccs = np.mean(comprehensive_mfccs, axis = 0)
    return comprehensive_mfccs



def record_audio_train():
    Name = (input("Please Enter Your Name:"))
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("----------------------record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(numdevices):
            if audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                print("Input Device id " + str(i) + " - " +
                      audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("-------------------------------------------------------------")
        index = int(input())
        print("recording via index "+str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        print("recording started")
        Recordframes = []
        for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = Name+"-sample"+str(count)+".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()



def record_audio_test():

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ",
                  audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())
    print("recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    savedname = input("Please input saved wave filename: ")
    OUTPUT_FILENAME = savedname + ".wav"
    WAVE_OUTPUT_FILENAME = "testing_set/" + OUTPUT_FILENAME
    trainedfilelist = open("testing_set_addition.txt", 'w')
    for fname in os.listdir("testing_set/"):
        if fname.endswith('.wav'):
            trainedfilelist.write(fname + "\n")
    trainedfilelist.write(OUTPUT_FILENAME +"\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()


def train_model():

    source = "training_set/"
    dest = "trained_models/"
    train_file = "training_set_addition.txt"
    
    file_paths = open(train_file, 'r')
    features = []

    for path in file_paths:
        path = path.strip()
        class_label = path.split("-")[0]
        print(path)
        data = extract_features(source + path)
        features.append([data, class_label])

    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    print(featuresdf.head())
    print(featuresdf.iloc[0]['feature'])


def test_model():

    source = "testing_set/"
    modelpath = "trained_models/"
    test_file = "testing_set_addition.txt"
    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]

    # Read the test directory and get the list of test audio files
    for path in file_paths:

        try:
            path = path.strip()
            sr, audio = read(source + path)
            vector = extract_features(audio)
        except Exception as e:
            print(e)
            print("error: " + path + " not found")
            continue

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        print(path," detected as - ", speakers[winner])
        time.sleep(1.0)


while True:
    choice = int(input(
        "\n 1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n"))
    if(choice == 1):
        record_audio_train()
    elif(choice == 2):
        train_model()
    elif(choice == 3):
        record_audio_test()
    elif(choice == 4):
        test_model()
    if(choice > 4):
        exit()
