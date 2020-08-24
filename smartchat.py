import os
from typing import List

import numpy as np
from tensorflow.keras import layers, preprocessing
import pathlib

import generate_multiple_answers as gma
import utils
from bigrams import Bigramer
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data
import warnings


class Chatbot:
    def __init__(self, model_filename, tokenizer, mle_model, bigramer, max_len_questions, max_len_answers, strategy):
        self._tokenizer = tokenizer
        self._tokenizer_index_to_word = {index: word for (word, index) in self._tokenizer.word_index.items()}

        self._mle_model = mle_model
        self._bigramer = bigramer
        self._max_len_questions = max_len_questions
        self._max_len_answers = max_len_answers

        model_data = utils.load_keras_model(model_filename)
        _, self._encoder_inputs, self._encoder_states, self._decoder_inputs, self._decoder_embedding, self._decoder_lstm, self._decoder_dense = model_data

        self._enc_model, self._dec_model = utils.make_inference_models(self._encoder_inputs, self._encoder_states,
                                                                       self._decoder_inputs, self._decoder_embedding,
                                                                       self._decoder_lstm, self._decoder_dense)
        self._strategy = strategy

    @classmethod
    def load_setup(cls, setup_filename, alternative_model_file=None):
        """
        Creates a chatbot from setup bundle.
        Preferred for production.

        :param setup_filename: path to json file with setup info.
        :param alternative_model_file: alternative file to load TF model from (instead of the one from setup .json file)
        :return: Chatbot initialized from setup bundle.
        """
        setup = utils.load_and_unjson(setup_filename)
        setup_directory = pathlib.Path(setup_filename).parent
        tokenizer = utils.load_and_unpickle(setup_directory / setup['tokenizer'])
        mle_model = utils.load_and_unpickle(setup_directory / setup['mle_model'])
        bigramer = Bigramer(dict=utils.load_and_unjson(setup_directory / setup['bigramer']))

        model_filename = alternative_model_file if alternative_model_file is not None \
            else setup_directory / setup['model']
        return cls(model_filename,
                   tokenizer,
                   mle_model,
                   bigramer,
                   setup['max_len_questions'],
                   setup['max_len_answers'],
                   setup['strategy'])

    @classmethod
    def load_from_params(cls):
        """
        Creates a chatbot from params.py.
        Good for development, not for production.

        :return: Chatbot initialized from params.py.
        """

        import params
        # load data
        questions, answers = load_data(params.data_file_directory, params.files, params.encoding)
        bigramer = Bigramer(params.bigramer)

        # prepare data manipulators
        VOCAB_SIZE = params.vocab_size
        tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, params.unknown_token)
        tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer,
                                                              questions,
                                                              answers)

        # prepare data
        prepared_data = prepare_data(tokenized_questions, tokenized_answers)
        max_len_questions, max_len_answers, *_ = prepared_data

        # mle_model
        reversed_tokenizer_word_dict = {index: word for (word, index) in tokenizer.word_index.items()}
        mle_model = utils.fit_mle_model(tokenized_answers, reversed_tokenizer_word_dict)

        # load model
        model_data = utils.load_keras_model(params.model)
        _, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = model_data

        return cls(params.model, tokenizer, mle_model, bigramer, max_len_questions, max_len_answers, params.strategy)

    def _empty_sequence_factory(self):
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = self._tokenizer.word_index['start']
        return empty_target_seq

    def _str_to_tokens(self, sentence):
        """
        Turn string into list of ints (indices of words in tokenizer)

        :param sentence: string with a full sentence
        :return: list of ints (indices of words in tokenizer) padded with zeros
        """
        tokens_list = self._tokenizer.texts_to_sequences([sentence])
        return preprocessing.sequence.pad_sequences(tokens_list, maxlen=self._max_len_questions, padding='post')

    def _tokens_to_str(self, tokens: List[str]):
        """
        Turn list of strings (tokens) into a user-friendly sentence.

        :param tokens: list of strings
        :return: pretty sentence as a string
        """
        if len(tokens) == 0:
            return 'Hmm.'  # when nothing useful was returned by the chatbot

        end_character = '?' if tokens[0] in ('where', 'what', 'who', 'is', 'are', 'how', 'when') else '.'
        tokens[0] = tokens[0].capitalize()
        tokens = ['I' if token == 'i' else token for token in tokens]
        return ' '.join(tokens) + end_character

    def _give_n_decoded(self, tokens, n=1, strip_start_end=True):
        """
        Give N beam-search answers from a chatbot as a list of lists of strings (tokens).

        :param tokens: list of ints (indices of words in tokenizer)
        :param n: number of anwers
        :param strip_start_end: if 'start' and 'end' tokens should be removed from each answer or not
        :return: list of answers (list of string tokens)
        """
        end_index = self._tokenizer.word_index['end']
        empty_target_seq = self._empty_sequence_factory()
        states_values = self._enc_model.predict(tokens)

        predictions, _ = gma.beam_search(states_values, empty_target_seq, self._dec_model, end_index, k=n)
        decoded = []
        for prediction in predictions:
            if strip_start_end:
                decoded.append([self._tokenizer_index_to_word.get(i, 'UNK') for i in prediction[1:-1]])
            else:
                decoded.append(
                    ['start'] + [self._tokenizer_index_to_word.get(i, 'UNK') for i in prediction[1:-1]] + ['end'])
        return decoded

    def _greedy(self, tokens):
        empty_target_seq = self._empty_sequence_factory()
        states_values = self._enc_model.predict(tokens)

        stop_condition = False
        output_tokens = []
        while not stop_condition:
            dec_outputs, h, c = self._dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = self._tokenizer_index_to_word.get(sampled_word_index, 'UNK')
            output_tokens.append(sampled_word)

            if sampled_word == 'end' or len(output_tokens) > self._max_len_answers:
                stop_condition = True

            empty_target_seq = self._empty_sequence_factory()
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return output_tokens[:-1]  # skip 'end' token

    def _heuristic(self, tokens):
        decoded = self._give_n_decoded(tokens, n=10, strip_start_end=False)

        # remove as many UNKs as possible
        bigrammed = [self._bigramer.replace_unks(words) for words in decoded]
        without_unks = [words for words in bigrammed if 'UNK' not in words]

        best = None
        if len(without_unks) == 0:
            # jeśli każdy zawiera co najmniej jednego UNK, to wybieramy najlepszy i usuwamy z niego UNKi
            best = utils.choose_best(bigrammed, self._mle_model)
            best = self._bigramer.strip_unks(best)
        else:
            # w przeciwnym razie wybieramy najlepszego spośród tych bez UNKów
            best = utils.choose_best(without_unks, self._mle_model)

        best = best[1:-1]  # strip 'start' and 'end'
        return best

    def one_chat(self, input, as_tokens=False):
        # flow: input (str) -> indexes in tokenizer ([int]) -> list of detokenized prediction ([str]) -> output (str)

        # list of tokens (integers)
        tokenized_input = self._str_to_tokens(input)

        # list of tokens (strings)
        if self._strategy == 'heuristic':
            tokenized_answer = self._heuristic(tokenized_input)
        else:  # greedy strategy is default
            tokenized_answer = self._greedy(tokenized_input)

        if as_tokens:
            return tokenized_answer
        else:
            # lisf of tokens (strings)
            tokens_without_unknowns = self._bigramer.replace_unks(tokenized_answer, starting_unknown='Hmm')
            tokens_without_unknowns = self._bigramer.strip_unks(tokens_without_unknowns)

            # output string
            stringified_output = self._tokens_to_str(tokens_without_unknowns)
            return stringified_output

    def chat(self, ntimes=10, as_tokens=False):
        for _ in range(ntimes):
            user_input = input('Enter question: ')
            answer = self.one_chat(user_input, as_tokens=as_tokens)
            print('Chatbot:', answer)

    def chat_forever(self, as_tokens=False):
        while True:
            user_input = input('Enter question: ')
            answer = self.one_chat(user_input, as_tokens=as_tokens)
            print('Chatbot:', answer)


def main():
    setups_dir = 'setups/'
    warnings.simplefilter('ignore')
    print('Enter name of a setup to start. Available setups:')

    available_setups = os.listdir(setups_dir)
    for i, setup_name in enumerate(available_setups, start=1):
        print(f'{i}. {setup_name}')
    choice = input('>')

    bot = None
    try:
        setup_path = pathlib.Path(setups_dir) / choice / f'{choice}.json'
        bot = Chatbot.load_setup(setup_path)
    except:
        print('Something went wrong when loading a setup for chatbot. Are you sure you entered a proper setup name?')
        exit(1)

    try:
        print(f'Loaded setup "{choice}". Ctrl+C to stop the conversation and exit. ')
        bot.chat_forever()
    except KeyboardInterrupt:
        print('Conversation has ended')
        exit(0)
    except:
        print('Something went wrong with the chatbot, exiting... ')


if __name__ == '__main__':
    main()
