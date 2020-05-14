import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
from tensorflow.python.client import device_lib
import pickle
from tensorflow.keras import layers, activations, models, preprocessing, utils
from gensim.models import Word2Vec
import re

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    answers_file = open("prepare_data/output_files/train2_good.from", encoding="utf-8")
    questions_file = open("prepare_data/output_files/train2_good.to", encoding="utf-8")
    answers_raw = answers_file.readlines()
    questions_raw = questions_file.readlines()
    answers = list()
    for answer in answers_raw:
        answers.append('<START> ' + answer + ' <END>')
    questions = questions_raw

    tokenizer = preprocessing.text.Tokenizer(num_words=15000)
    tokenizer.fit_on_texts(questions + answers)
    VOCAB_SIZE = 15000 # len(tokenizer.word_index) + 1
    print('VOCAB SIZE : {}'.format(VOCAB_SIZE))

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    tokenized_answers = tokenizer.texts_to_sequences(answers)

    tokenized_questions_good = []
    tokenized_answers_good = []

    for tokenized_question, tokenized_answer in zip(tokenized_questions, tokenized_answers):
        if (len(tokenized_answer) > 7 and len(tokenized_question) > 7):
            tokenized_questions_good.append(tokenized_question)
            tokenized_answers_good.append(tokenized_answer)

    tokenized_questions = tokenized_questions_good
    tokenized_answers = tokenized_answers_good

    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions,
                                                            padding='post')
    encoder_input_data = np.array(padded_questions)
    print(encoder_input_data.shape, maxlen_questions)

    # decoder_input_data

    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answers)
    print(decoder_input_data.shape, maxlen_answers)

    # decoder_output_data
    tokenized_answers = tokenized_answers_good
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
    # onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)

    p = list(filter(lambda x: x[1][1] == 0, enumerate(padded_answers)))

    decoder_output_data = np.array(padded_answers)
    print(decoder_output_data.shape)

    new_model = load_model('weights.best_200_1.hdf5')


    # fit the model
    checkpoint = ModelCheckpoint('weights.best_200_2', verbose=1)
    callbacks_list = [checkpoint]
    new_model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list, batch_size=1,
              epochs=1)
    new_model.save('model_200_2.h5')