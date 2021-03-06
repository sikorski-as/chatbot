import nltk

import params
import utils
import pathlib
import shutil
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data

if __name__ == '__main__':
    #
    # setup file data
    #
    setup_name = params.files  # tu można dać w sumie cokolwiek innego, to tylko nazwa
    model_filename = f'{setup_name}.hdf5'
    setup = {
        'model': model_filename,
        'tokenizer': f'{setup_name}.tokenizer',
        'bigramer': f'{setup_name}.bigramer',
        'mle_model': f'{setup_name}.mle',
        'max_len_questions': 0,
        'max_len_answers': 0,
        'strategy': 'greedy',
        'vocab_size': params.vocab_size,
        'encoding': params.encoding,
        'unknown_token': params.unknown_token
    }

    setup_path = pathlib.Path(f'setups/{setup_name}')
    setup_path.mkdir(parents=True, exist_ok=True)
    model_file_path = shutil.copy(params.model, setup_path / model_filename)

    #
    # load raw data
    #
    print('Loading data!')
    questions, answers = load_data(params.data_file_directory, params.files, params.encoding)
    print('Data loaded!')

    #
    # load model
    #
    # model_data = utils.load_keras_model(params.model)
    # _, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = model_data

    #
    # create and save tokenizer
    #
    print('Creating tokenizer!')
    tokenizer = create_tokenizer(questions + answers, params.vocab_size, params.unknown_token)
    reversed_tokenizer_word_dict = {index: text for text, index in tokenizer.word_index.items()}
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)
    prepared_data = prepare_data(tokenized_questions, tokenized_answers)
    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = prepared_data
    setup['max_len_questions'] = max_len_questions
    setup['max_len_answers'] = max_len_answers

    utils.pickle_and_save(tokenizer, setup_path / setup['tokenizer'])
    print('Tokenizer created and saved!')

    #
    # create and save mle_model
    #
    print('Creating MLE model!')
    mle_model = utils.fit_mle_model(tokenized_answers, reversed_tokenizer_word_dict)
    utils.pickle_and_save(mle_model, setup_path / setup['mle_model'])
    print('MLE model created and saved!')

    #
    # create and save bigramer
    #
    print('Creating bigramer!')
    bigrams_frequency = {}
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
            bigrams_frequency[first] = {}
            bigrams_frequency[first][second] = 1

    bigrams_frequency_best = {}
    for first in bigrams_frequency:
        bigrams_frequency_best[first] = \
            sorted(bigrams_frequency[first].items(), key=lambda x: x[1], reverse=True)[0][0]

    utils.json_and_save(bigrams_frequency_best, setup_path / setup['bigramer'], nice=True)
    print('Bigramer created and saved!')

    #
    # save setup file
    #
    utils.json_and_save(setup, setup_path / f'{setup_name}.json', nice=True)
    print('Setup created!')
