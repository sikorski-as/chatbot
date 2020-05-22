from cmath import log

import nltk
import tensorflow as tf
from nltk.lm import Vocabulary, Laplace
from tensorflow_core.python.keras.saving.save import load_model
import numpy as np


def create_keras_model(vocab_size):
    encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_input")
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, mask_zero=True, name="encoder_embedding")(
        encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True, name="encoder_lstm")(
        encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_input")
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, name="decoder_embedding", mask_zero=True)(
        decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax, name="decoder_dense")
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense


def load_keras_model(file):
    # model = load_model(file)
    model = load_model('model_test.h5')
    model.load_weights(file)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]

    decoder_inputs = model.input[1]  # input_2
    decoder_embedding = model.layers[3].output
    decoder_lstm = model.layers[5]
    decoder_dense = model.layers[6]

    return model, encoder_inputs, encoder_states, decoder_inputs, \
           decoder_embedding, decoder_lstm, decoder_dense


def fit_mle_model(text, text_dict):
    # text dict key: index value: text, nie ma w tokenizer domyslnie trzeba odwrocic slownik
    model = Laplace(2)
    tokenized_text = [[text_dict[index] for index in sentence] for sentence in text]
    train_data = [nltk.bigrams(t) for t in tokenized_text]
    words = [word for sentence in tokenized_text for word in sentence]
    vocab = Vocabulary(words)
    model.fit(train_data, vocab)
    return model


def calculate_mle(decoded_translations: list, model) -> list:
    test_data = [nltk.bigrams(t) for t in decoded_translations]
    results = []
    for test in test_data:
        score = 0
        for ngram in test:
            score = score + log(model.score(ngram[-1], ngram[:-1]))
        results.append(score)
    return results


def calculate_perplexity(decoded_translations: list, model) -> list:
    test_data = [nltk.bigrams(t) for t in decoded_translations]
    results = []
    for test in test_data:
        score = model.perplexity(test)
        results.append(score)
    return results


def choose_best(decoded_translations: list, model):
    # wybiera odpowiedz o najwiekszym prawdopodobienstwie
    results = calculate_perplexity(decoded_translations, model)
    # print(results)
    best_index = np.argsort(results)[0]
    # print(best_index)
    # print(decoded_translations[best_index])
    return decoded_translations[best_index]


if __name__ == '__main__':
    load_keras_model('checkpoints/train2/cp-0004.hdf5')
