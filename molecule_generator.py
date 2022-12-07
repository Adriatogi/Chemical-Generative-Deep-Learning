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

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = Tokenizer.texts_to_sequences([seed_text])[0]
        #token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in Tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

seed = '!!!!!!!!!!!!!!?'
model = keras.models.load_model('hybrid.h5')
generate_text(seed, 35, model, 50)
