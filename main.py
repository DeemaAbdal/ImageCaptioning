import os.path
from os import path
import requests
import json
import os
import keras
import keras.utils as image
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import json
from pickle import load, dump
import tensorflow as tf
import string

if path.exists('/content/image-captioning') == False:
  os.mkdir('/content/image-captioning')

os.chdir('/content/image-captioning')
if path.exists('/content/image-captioning/sample_images') == False:
  os.mkdir('/content/image-captioning/sample_images')

def feature_extractions(directory):
    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = {}
    for f in os.listdir(directory):
        filename = r'/content/image-captioning/sample_images/img1.jpg'
        identifier = f.split('.')[0]
        image = tf.keras.utils.load_img(r'/content/image-captioning/sample_images/img2.jpg', target_size=(224,224))
        arr = keras.utils.img_to_array(image, dtype=np.float32)
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = keras.applications.vgg16.preprocess_input(arr)
        feature = model.predict(arr, verbose=0)
        features[identifier] = feature
    return(features)
def sample_caption(model, tokenizer, max_length, vocab_size, feature):
    caption = "<startseq>"
    while 1:
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))
        pred_Y = model.predict([feature, padded])[0,-1,:]
        next_word = tokenizer.index_word[pred_Y.argmax()]
        caption = caption + ' ' + next_word
        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break
    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')
    return(caption)
with open('/content/image-captioning/File1.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)
model = keras.models.load_model("/content/image-captioning/File2.h5")
vocab_size = tokenizer.num_words
max_length = 37
features = feature_extractions("/content/image-captioning/sample_images")
for i, filename in enumerate(features.keys()):
  plt.figure
  caption = sample_caption(model, tokenizer, max_length, vocab_size, features[filename])
  img = tf.keras.utils.load_img("/content/image-captioning/sample_images/img2.jpg")
  plt.imshow(img)
print (caption)
