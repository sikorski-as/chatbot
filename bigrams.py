import json

import nltk

import params
from data import load_data, create_tokenizer, tokenize_q_a


class Bigramer:
    def __init__(self, file=None, dict=None):
        self.bigrams_frequency = {}
        if file is not None:
            with open(file, "r") as f:
                self.bigrams_frequency = json.loads(f.read())
        elif dict is not None:
            self.bigrams_frequency = dict

    def give_word(self, word):
        return self.bigrams_frequency.get(word, "UNK")

    def fill_unks(self, sentence):
        tokens = sentence.split(' ')
        return self.replace_unks(tokens)

    def replace_unks(self, tokens, starting_unknown=None):
        for i in range(len(tokens)):
            if tokens[i] == 'UNK':
                if starting_unknown is not None and i == 0:
                    tokens[i] = starting_unknown
                elif tokens[i - 1] != 'UNK':
                    tokens[i] = self.give_word(tokens[i - 1])
        return tokens

    def strip_unks(self, tokens):
        return [token for token in tokens if token != 'UNK']


if __name__ == '__main__':
    questions, answers = load_data(params.data_file_directory, params.files, params.encoding)
    tokenizer = create_tokenizer(questions + answers, None, None)
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)
    reversed_tokenizer_word_dict = {index: text for text, index in tokenizer.word_index.items()}
    bigrams_frequency = dict()
    flatten = lambda l: [item for sublist in l for item in sublist]
    tokenized_text = [[reversed_tokenizer_word_dict[index] for index in sentence] for sentence in
                      tokenized_questions + tokenized_answers]
    bigrams = flatten([list(nltk.bigrams(t)) for t in tokenized_text])
    for (first, second) in bigrams:
        if first in bigrams_frequency:
            if second in bigrams_frequency[first]:
                bigrams_frequency[first][second] += 1
            else:
                bigrams_frequency[first][second] = 1
        else:
            bigrams_frequency[first] = dict()
            bigrams_frequency[first][second] = 1

    try:
        bigrams_frequency_best = dict()
        for first in bigrams_frequency:
            bigrams_frequency_best[first] = sorted(bigrams_frequency[first].items(), key=lambda x: x[1], reverse=True)[0][0]
        with open(f"{params.files}_bigramer", "a") as b:
            b.write(json.dumps(bigrams_frequency_best))
        print('successfully created a bigramer')
    except:
        print('bigrammer not created!')
        raise
