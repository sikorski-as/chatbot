from keras_preprocessing.text import Tokenizer
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.models import Model, load_model
import numpy as np

from data import load_data, create_tokenizer, tokenize_q_a, prepare_data


def str_to_tokens(tokenizer: Tokenizer, sentence: str, max_len_questions):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=max_len_questions, padding='post')


def make_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense):
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = layers.Input(shape=(200,), name="input_h")
    decoder_state_input_c = layers.Input(shape=(200,), name="input_c")

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


if __name__ == '__main__':
    questions, answers = load_data("prepare_data/output_files", "preprocessed_train")
    VOCAB_SIZE = 15001
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, 'UNK')
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    model = load_model('model_test.h5')

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]

    decoder_inputs = model.input[1]  # input_2
    decoder_embedding = model.layers[3].output
    decoder_lstm = model.layers[5]
    decoder_dense = model.layers[6]

    enc_model, dec_model = make_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense)

    for _ in range(10):
        states_values = enc_model.predict(str_to_tokens(tokenizer, input('Enter question : '), max_len_questions))
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

            if sampled_word == 'end' or len(decoded_translation.split()) > max_len_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        print(decoded_translation)
