from typing import List

import numpy as np
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.models import Model

import params
import utils
from bigrams import Bigramer
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data


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


class BaseChatbot:
    def __init__(self):
        # load data
        self._questions, self._answers = load_data(params.data_file_directory, params.files, params.encoding)
        self._bigramer = Bigramer(params.bigramer)

        # prepare data manipulators
        self._VOCAB_SIZE = params.vocab_size
        self._tokenizer = create_tokenizer(self._questions + self._answers, self._VOCAB_SIZE, params.unknown_token)
        self._tokenized_questions, self._tokenized_answers = tokenize_q_a(self._tokenizer,
                                                                          self._questions,
                                                                          self._answers)

        self._tokenizer_index_to_word = {index: word for (word, index) in self._tokenizer.word_index.items()}

        # prepare data
        prepared_data = prepare_data(self._tokenized_questions, self._tokenized_answers)
        self._max_len_questions, self._max_len_answers, self._encoder_input_data, self._decoder_input_data, self._decoder_output_data = prepared_data

        # load model
        model_data = utils.load_keras_model(params.model)
        _, self._encoder_inputs, self._encoder_states, self._decoder_inputs, self._decoder_embedding, self._decoder_lstm, self._decoder_dense = model_data

        # prepare encoder/decoder models
        self._enc_model, self._dec_model = make_inference_models(self._encoder_inputs, self._encoder_states,
                                                                 self._decoder_inputs, self._decoder_embedding,
                                                                 self._decoder_lstm, self._decoder_dense)

    def _empty_sequence_factory(self):
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = self._tokenizer.word_index['start']
        return empty_target_seq

    def _str_to_tokens(self, sentence):
        # words = sentence.lower().split()
        # tokens_list = list()
        # for word in words:
        #     tokens_list.append(self._tokenizer.word_index.get(word, None))
        tokens_list = self._tokenizer.texts_to_sequences([sentence])
        # print(tokens_list, tokens_list2)
        return preprocessing.sequence.pad_sequences(tokens_list, maxlen=self._max_len_questions, padding='post')

    def _tokens_to_str(self, tokens):
        return ' '.join(tokens)

    def _heuristic(self, tokens):
        empty_target_seq = self._empty_sequence_factory()
        states_values = self._enc_model.predict(tokens)

        stop_condition = False
        output_tokens = []
        while not stop_condition:
            dec_outputs, h, c = self._dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = self._tokenizer_index_to_word.get(sampled_word_index, params.unknown_token)
            output_tokens.append(sampled_word)

            if sampled_word == 'end' or len(output_tokens) > self._max_len_answers:
                stop_condition = True

            empty_target_seq = self._empty_sequence_factory()
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return output_tokens[:-1]  # skip 'end' token

    def one_chat(self, input, as_tokens=False):
        # flow: input (str) -> indexes in tokenizer ([int]) -> list of detokenized prediction ([str]) -> output (str)

        # list of tokens (integers)
        tokenized_input = self._str_to_tokens(input)

        # list of tokens (strings)
        tokenized_answer = self._heuristic(tokenized_input)
        if as_tokens:
            return tokenized_answer
        else:
            # lisf of tokens (strings)
            tokens_without_unknowns = self._bigramer.replace_unks(tokenized_answer, starting_unknown='Hmm')

            # output string
            stringified_output = self._tokens_to_str(tokens_without_unknowns)
            return stringified_output

    def chat(self):
        for _ in range(10):
            user_input = input('Enter question: ')
            answer = self.one_chat(user_input)
            print('Chatbot:', answer)


class Chatbot(BaseChatbot):
    def _tokens_to_str(self, tokens: List[str]):
        end_character = '?' if tokens[0] in ('where', 'what', 'who', 'is', 'are', 'how', 'when') else '.'
        tokens[0] = tokens[0].capitalize()
        tokens = ['I' if token == 'i' else token for token in tokens]
        return ' '.join(tokens) + end_character


def main():
    bot = Chatbot()
    bot.chat()


if __name__ == '__main__':
    main()
