import numpy as np
import random
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

words = re.findall(r'[а-яё\-]+', text)

vocab = sorted(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Всего слов: {len(words)}")
print(f"Уникальных слов: {vocab_size}")

max_sequence_len = 10
step = 1
sequences = []
next_words = []

for i in range(0, len(words) - max_sequence_len, step):
    sequences.append(words[i:i+max_sequence_len])
    next_words.append(words[i+max_sequence_len])

# One-hot кодирование
X = np.zeros((len(sequences), max_sequence_len, vocab_size), dtype=bool)
y = np.zeros((len(sequences), vocab_size), dtype=bool)

for i, seq in enumerate(sequences):
    for t, word in enumerate(seq):
        X[i, t, word_to_idx[word]] = 1
    y[i, word_to_idx[next_words[i]]] = 1

print(f"Входная матрица X: {X.shape}")
print(f"Выходная матрица y: {y.shape}")

model = Sequential()
model.add(LSTM(128, input_shape=(max_sequence_len, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_words, length, temperature):
    # если на вход дана строка, разобьём на слова
    if isinstance(seed_words, str):
        seed_words = re.findall(r'[а-яё\-]+', seed_words.lower())
    
    # обрезаем или дополняем до max_sequence_len
    if len(seed_words) > max_sequence_len:
        seed_words = seed_words[-max_sequence_len:]
    elif len(seed_words) < max_sequence_len:
        seed_words = [''] * (max_sequence_len - len(seed_words)) + seed_words
    
    generated = seed_words.copy()
    for _ in range(length):
        x_pred = np.zeros((1, max_sequence_len, vocab_size))
        for t, word in enumerate(seed_words):
            if word in word_to_idx:
                x_pred[0, t, word_to_idx[word]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_word = idx_to_word[next_idx]
        
        generated.append(next_word)
        seed_words = seed_words[1:] + [next_word]
    
    return ' '.join(generated)

print("\nНачинаем обучение...")
model.fit(X, y, batch_size=128, epochs=50, verbose=1)

print("\nГенерируем текст...")
result = generate_text(
    seed_words="жил был колобок",
    length=1500,
    temperature=0.6
)

with open('../result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(result)

print("Готово. Результат сохранён в result/gen.txt")
