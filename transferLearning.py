
import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import random as rn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array, savetxt, asarray, save
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding, Dropout
from keras.utils import np_utils
#from google.colab import drive
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import GRU
drive.mount('/content/drive')
import keras_tuner
from keras_tuner import HyperModel

DATA_PATH = "data/alzheimersdata.txt"
NEW_DATA = "data/TL_processed_data.txt"
TRAIN_DATA = "data/TL_train_data.txt"
TEST_DATA = "data/TL_test_data.txt"
VAL_DATA = "data/TL_val_data.txt"

"""Filter Dataset Into Smaller Segments"""
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))
print(file_new.shape)

"""Find Maximum Sequence Length"""

file = open(NEW_DATA)
max_seq_len = int(len(max(file,key=len)))
print ("Max Sequence Length: ", max_seq_len)

"""Define Functions"""

def read(fileName):
        fileObj = open(fileName, "r")
        words = fileObj.read().splitlines()
        fileObj.close()
        return words

def padFile(fileName):
  temp = read(fileName)
  preprocessed_pad_text = [['?'] + list(i) for i in temp]
  print("Sample 1: ", preprocessed_pad_text[0])
  print("Sample 2: ", preprocessed_pad_text[1])
  print("Sample 3: ", preprocessed_pad_text[2])
  print("Sample 4: ", preprocessed_pad_text[3])
  print("Sample 5: ", preprocessed_pad_text[4])

  padded_text = pad_sequences(preprocessed_pad_text, dtype=object, maxlen=max_seq_len+1, padding="post", value="!")
  # front_pad = [['?'] + i for i in padded_text]
  #print("Padded Text Arrays: ", padded_text)

  var = ["".join(i) for i in padded_text]
  print("Padded Strings: ", var[0:5])
  #print(var)
  # with open('/content/drive/MyDrive/CS230/padded_processed_data.txt', "w") as output:
  #   for x in var:
  #     output.write(y)
  # np.array(var)
  # np.savetxt('/content/drive/MyDrive/CS230/padded_processed_data.txt', var, fmt ='%s', newline='')
  return var


var = padFile(NEW_DATA)
var = np.array(var)

"""Load & Save Datasets"""

#init random seed
seed = 1
np.random.seed(seed)
#split data into train/test
full_train, test = train_test_split(np.array(var), test_size=0.2, random_state=seed)
# full_train, test = train_test_split(np.array(var), test_size=0.25, random_state=seed)

#split full train set into smaller train set and validation (dev) set
# np.savetxt('/content/drive/MyDrive/CS230/full_train_data.txt', full_train, fmt ='%s', newline='')
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)
np.savetxt(TRAIN_DATA, train, fmt ='%s', newline='')
np.savetxt(TEST_DATA, test, fmt ='%s', newline='')
np.savetxt(VAL_DATA, val, fmt='%s', newline='')
print("Sample:", train[seed])
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

"""Train Data Processing"""

#Load Data (optional: only if previous cells are not run and data is saved already)
# train = pd.read_fwf('/content/drive/MyDrive/CS230/train_data.txt')

#Concatenate Data
def concatenate(data):
    #res = ''
    print(len(data))
    #for count, word in enumerate(data):
        #if count %1000 == 0:
            #print(count)
            #pass
        # print(word)
        #res += word
    res = ''.join(data)
    return res
train = concatenate(train)
print(train[0:100])

#Tokenize Data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, lower=False)
tokenizer.fit_on_texts(train)
new_train = tokenizer.texts_to_sequences(train)
print(new_train[0:100])
print(train[0:100])

#Print Data Breakdown
n_chars = len(train)
n_vocab = len(list(set(train)))
print("# of Unique Characters:", n_chars)
print("# of Total Characters:", n_vocab)
n_chars = len(new_train)
n_vocab = len(set(train))
print("# of Unique Characters:", n_chars)
print("# of Total Characters:", n_vocab)

#N-Grams Sequence
seqLen = 15
stepSize = 1
input_chars = []
next_char = []

for i in range(0, len(new_train) - seqLen, stepSize):
  input_chars.append(new_train[i : i + seqLen])
  next_char.append(new_train[i + seqLen])
for i in range(5):
  print("Input Sequence:", input_chars[i])
  print("Next Character Prediction:", next_char[i])
#Assemble Train Datasets
x_train = np.array(input_chars)
x_train.flatten()
y_train = np.array(next_char)
y_train_2 = np_utils.to_categorical(y_train)

print(f"x_train shape {x_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"x_train {x_train[0:5]}")
print(f"y_train {y_train[0:5]}")


transferLearned_BaselineLSTM = keras.models.load_model('baseline_lstm.h5')
transferLearned_BaselineLSTM.layers[0].trainable = False
transferLearned_BaselineLSTM.layers[1].trainable = False
transferLearned_BaslineLSTM.fit(ADD X TRAIN, ADD Y TRAIN, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early])


transferLearned_Hybrid = keras.models.load_model('hybrid.h5')
transferLearned_Hybrid.layers[0].trainable = False
transferLearned_Hybrid.layers[1].trainable = False
transferLearned_Hybrid.fit(ADD X TRAIN, ADD Y TRAIN, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early])
p
