from keras_preprocessing.text import Tokenizer
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.models import Model, load_model
import numpy as np

import params
import utils
from bigrams import Bigramer
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data
import generate_multiple_answers as gma

# bigramer = Bigramer(params.bigramer)


def str_to_tokens(tokenizer: Tokenizer, sentence: str, max_len_questions):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        token = tokenizer.word_index.get(word, tokenizer.word_index['UNK'])
        if token < params.vocab_size:
            tokens_list.append(token)
        else:
            tokens_list.append(tokenizer.word_index['UNK'])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=max_len_questions, padding='post')


def make_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm,
                          decoder_dense):
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = layers.Input(shape=(params.decoder_state_input_h_size,), name="input_h")
    decoder_state_input_c = layers.Input(shape=(params.decoder_state_input_h_size,), name="input_c")

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def choose_greedy(empty_target_seq, states_values):
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

        if sampled_word == 'end' or len(decoded_translation.split()) > max_len_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)


def choose_beam(states_values, empty_target_seq, dec_model, end_index):
    predictions, _ = gma.beam_search(states_values, empty_target_seq, dec_model, end_index)

    for prediction in predictions:
        decoded_translation = ''
        for sampled_word_index in prediction[1:]:
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
        print(decoded_translation)
        # print(bigramer.fill_unks(decoded_translation))
    print()


if __name__ == '__main__':
    questions, answers = load_data(params.data_file_directory, params.files, params.encoding)
    VOCAB_SIZE = params.vocab_size
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, params.unknown_token)
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    prepared_data = prepare_data(tokenized_questions, tokenized_answers)
    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = prepared_data

    model_data = utils.load_keras_model(params.model)
    _, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = model_data

    enc_model, dec_model = make_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding,
                                                 decoder_lstm, decoder_dense)

    end_index = tokenizer.word_index['end']
    for _ in range(10):
        states_values = enc_model.predict(str_to_tokens(tokenizer, input('Enter question : '), max_len_questions))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        choose_greedy(empty_target_seq, states_values)
        # choose_beam(states_values, empty_target_seq, dec_model, end_index)
