import numpy as np
import tensorflow as tf
import random
import re
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint

with open(r'C:\Users\vikto\Desktop\b81_7\b81_7-main\src\input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokens = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)

vocabulary = sorted(list(set(tokens)))
word_to_indices = dict((word, i) for i, word in enumerate(vocabulary))
indices_to_word = dict((i, word) for i, word in enumerate(vocabulary))

max_length = 15
step = 1

sentences = []
next_words = []

for i in range(0, len(tokens) - max_length, step):
    sentences.append(tokens[i:i + max_length])
    next_words.append(tokens[i + max_length])

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length_in_words, diversity):
    start_index = random.randint(0, len(tokens) - max_length - 1)
    generated_words = tokens[start_index:start_index + max_length].copy()
    generated_text = ' '.join(generated_words)
    
    for i in range(length_in_words):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, word in enumerate(generated_words):
            x_pred[0, t, word_to_indices[word]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = indices_to_word[next_index]
        
        generated_text += ' ' + next_word
        generated_words.append(next_word)
        generated_words.pop(0)
    
    return generated_text

model.fit(X, y, batch_size=128, epochs=50)

result_text = ""
for temp in [0.2]:
    result_text += generate_text(1000, temp) + "\n\n"

with open(r'C:\Users\vikto\Desktop\b81_7\b81_7-main\result\gen.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)

print(generate_text(1000, 0.6))
