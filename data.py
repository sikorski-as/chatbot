import pickle
import string

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_data(directory: string, filename: string) -> (list, list):
    with open(f"{directory}/{filename}.from", encoding="utf-8") as answers_file,\
            open(f"{directory}/{filename}.to", encoding="utf-8") as questions_file:
        answers_raw = answers_file.readlines()
        questions_raw = questions_file.readlines()
        answers = list()
        for answer in answers_raw:
            answers.append('<START> ' + answer + ' <END>')
        questions = questions_raw
        return questions, answers


def create_tokenizer(sequences: list, num_words: int, oov_token: string):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(sequences)
    return tokenizer


def save_tokenizer(filename: string, tokenizer: Tokenizer):
    with open(f"{filename}.pickle", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(filename: string) -> Tokenizer:
    with open(f"{filename}.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def tokenize_q_a(tokenizer: Tokenizer, questions: list, answers: list) -> (list, list):
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    tokenized_answers = tokenizer.texts_to_sequences(answers)

    tokenized_questions_good = []
    tokenized_answers_good = []

    for tokenized_question, tokenized_answer in zip(tokenized_questions, tokenized_answers):
        if len(tokenized_answer) > 0 and len(tokenized_question) > 0:
            tokenized_questions_good.append(tokenized_question)
            tokenized_answers_good.append(tokenized_answer)

    tokenized_questions = tokenized_questions_good
    tokenized_answers = tokenized_answers_good

    return tokenized_questions, tokenized_answers


def prepare_data(tokenized_questions, tokenized_answers) -> (int, int, list, list, list):
    # encoder_input_data
    max_len_questions = max([len(x) for x in tokenized_questions])
    padded_questions = pad_sequences(tokenized_questions, maxlen=max_len_questions, padding='post')
    encoder_input_data = np.array(padded_questions)

    # decoder_input_data
    max_len_answers = max([len(x) for x in tokenized_answers])
    padded_answers = pad_sequences(tokenized_answers, maxlen=max_len_answers, padding='post')
    decoder_input_data = np.array(padded_answers)

    # decoder_output_data
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = pad_sequences(tokenized_answers, maxlen=max_len_answers, padding='post')
    decoder_output_data = np.array(padded_answers)

    return max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data
