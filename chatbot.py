import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import pickle
from tensorflow.keras import layers, activations, models, preprocessing, utils
from gensim.models import Word2Vec
import re

from tensorflow.python.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    answers_file = open("prepare_data/output_files/test.from", encoding="utf-8")
    questions_file = open("prepare_data/output_files/test.to", encoding="utf-8")
    answers_raw = answers_file.readlines()
    questions_raw = questions_file.readlines()
    answers = list()
    for answer in answers_raw:
        answers.append('<START> ' + answer + ' <END>')
    questions = questions_raw

    tokenizer = preprocessing.text.Tokenizer(num_words=15001, oov_token='UNK')
    tokenizer.fit_on_texts(questions + answers)
    VOCAB_SIZE = 15001 # len(tokenizer.word_index) + 1
    print('VOCAB SIZE : {}'.format(VOCAB_SIZE))

    # vocab = []
    # for word in tokenizer.word_index:
    #     vocab.append(word)

    #     def tokenize(sentences):


    # tokens_list = []
    #     vocabulary = []
    #     for sentence in sentences:
    #         sentence = sentence.lower()
    #         sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    #         tokens = sentence.split()
    #         vocabulary += tokens
    #         tokens_list.append(tokens)
    #     return tokens_list, vocabulary

    # p = tokenize(questions + answers)
    # model = Word2Vec(p[0])

    # embedding_matrix = np.zeros((VOCAB_SIZE, 100))
    # for i in range(len(tokenizer.word_index)):
    #     embedding_matrix[i] = model[vocab[i]]

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    tokenized_answers = tokenizer.texts_to_sequences(answers)

    tokenized_questions_good = []
    tokenized_answers_good = []

    for tokenized_question, tokenized_answer in zip(tokenized_questions, tokenized_answers):
        if (len(tokenized_answer) > 0 and len(tokenized_question) > 0):
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
    # tokenized_answers = tokenized_answers_good
    tokenized_answers = tokenized_answers_good
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
    # onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)

    p = list(filter(lambda x: x[1][1] == 0, enumerate(padded_answers)))

    decoder_output_data = np.array(padded_answers)
    print(decoder_output_data.shape)

    encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_input")
    encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True, name="encoder_embedding")(encoder_inputs)
    # encoder_masking = layers.Masking(mask_value=0, name="encoder_masking")(encoder_embedding)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True, name="encoder_lstm")(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_input")
    decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, name="decoder_embedding", mask_zero=True)(decoder_inputs)
    # decoder_masking = tf.keras.layers.Masking(mask_value=0, name="decoder_masking")(decoder_embedding)
    decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax, name="decoder_dense")
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')

    model.summary()

    filepath = "weights.best_test.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)
    callbacks_list = [checkpoint]

    model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list, batch_size=6, epochs=1)
    model.save('model_test.h5')


    def make_inference_models():

        encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

        decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = tf.keras.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def str_to_tokens( sentence : str ):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append( tokenizer.word_index[ word ] )
        return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


    enc_model, dec_model = make_inference_models()

    for _ in range(10):
        states_values = enc_model.predict(str_to_tokens(input('Enter question : ')))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        print(decoded_translation)

