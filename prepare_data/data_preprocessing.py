from typing import List
from spellchecker import SpellChecker
from tensorflow.keras import preprocessing

MAX_LENGTH = 60

FORBIDDEN_WORDS = (
    'fuck', 'fucking', 'fuckin', 'cock', 'pussy', 'sex', 'idiot', 'dick', 'imbecile', 'faggot', 'moron', 'shit', 'porn')
UNNECESSARY_WORDS = (' the ', ' a ', ' an ', 'newlinechar')

spell_checker = SpellChecker()
spell_checker.word_frequency.load_text_file("unknown.txt")


def remove_unnecessary_words(line: str):
    for un_word in UNNECESSARY_WORDS:
        line = line.replace(un_word, ' ')
    return line


""" Jeśli prawda to znaczy że słowo jest okey"""
WORD_VALIDATORS = (
    # lambda word: re.fullmatch(r'[\D]+[0-9]+.*', word) is None,
    # wyrzuca słowa zawierające w środku cyfry, pewnie jakieś loginy
    # lambda word: re.search(r'o{3,10}?|s{3,6}?|u{3,6}?|z{3,6}?', word) is None,  # wyrzuca wyrazy z za dużą ilością znakó
    # lambda word: re.search(r'.*http.*', word) is None,  # wyrzuca linki
    lambda word: all(ord(ch) < 128 for ch in word),  # wyrzuca nie ascii
    lambda word: str.lower(word) not in FORBIDDEN_WORDS,  # wyrzuca brzydkie słowa
    # lambda word: len(spell_checker.known([word])) == 1
)


def correct_word(word) -> bool:
    for valid in WORD_VALIDATORS:
        if not valid(word):
            return False
    return True


def correct_lines(iterable, iterable2):
    for line, line2 in zip(iterable, iterable2):
        correct = True
        line = remove_unnecessary_words(line)
        line2 = remove_unnecessary_words(line2)
        for word, word2 in zip(line.split(), line2.split()):
            if not correct_word(word) or not correct_word(word2):
                correct = False
                break
        if correct:
            yield line, line2


def erase_wrong_comments(token_comments, token_replies, wrong_tokens: set):
    """
        Filtruje dane i zwraca indeksy poprawnych par
    """

    legit_pairs = ([], [])
    indexes = []
    for i, (token_comment, token_reply) in enumerate(zip(token_comments, token_replies)):
        correct = True
        if (len(token_comment) > MAX_LENGTH or len(token_reply) > MAX_LENGTH) or (
                len(token_comment) == 0 or len(token_reply) == 0):
            continue
        for token, token2 in zip(token_comment, token_reply):
            if token in wrong_tokens or token2 in wrong_tokens:
                correct = False
                break
        if correct:
            legit_pairs[0].append(token_comment)
            legit_pairs[1].append(token_reply)
            indexes.append(i)

    return legit_pairs, indexes


tokenizer = preprocessing.text.Tokenizer()


def tokenize_text(text: list):
    tokenizer.fit_on_texts(text)
    tokenized_text = tokenizer.texts_to_sequences(text)
    return tokenized_text


def save_comments(parent_lines: List[str], reply_lines: List[str], indexes: list):
    with open("output_files/preprocessed_train.from", mode="w", encoding="utf-8") as p_f:
        with open("output_files/preprocessed_train.to", mode="w", encoding="utf-8") as r_f:
            for index in indexes:
                p_f.write(parent_lines[index])
                r_f.write(reply_lines[index])


def preprocess(parent_comment_file="output_files/train.from", reply_comment_file="output_files/train.to"):
    parent_lines = []
    reply_lines = []
    with open(parent_comment_file, mode="r", encoding="utf8") as parent_file, \
            open(reply_comment_file, mode="r", encoding="utf8") as reply_file:
        for parent_line, reply_line in correct_lines(parent_file, reply_file):
            if len(parent_line.split()) <= MAX_LENGTH and len(reply_line.split()) <= MAX_LENGTH:
                parent_lines.append(parent_line)
                reply_lines.append(reply_line)

    tokenized_from = tokenize_text(parent_lines)
    tokenized_to = tokenize_text(reply_lines)
    print(len(tokenized_to), len(tokenized_from))
    """Wyciecie komentarzy ktorych tokeny sa niepoprawne"""
    comment_word_dict = tokenizer.word_index
    # wczytywanie niepoprawnych tokenów
    # with open("unknown.txt", mode="w", encoding="utf-8") as f:
    #
    wrong_tokens = set()
    for word, token_id in comment_word_dict.items():
        if spell_checker.known([word]) == set() and '\'' not in word:
            # print(word)
            # f.write(word + "\n")
            wrong_tokens.add(token_id)
    print(len(wrong_tokens))

    (tokenized_from, tokenized_to), indexes = erase_wrong_comments(tokenized_from, tokenized_to, wrong_tokens)

    print(len(tokenized_to), len(tokenized_from))
    print(indexes[:100])

    save_comments(parent_lines, reply_lines, indexes)


if __name__ == '__main__':
    preprocess()
