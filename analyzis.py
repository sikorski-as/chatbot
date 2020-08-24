import os
from collections import defaultdict
import params
import utils
from bigrams import Bigramer
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data
import conversation
import numpy as np

from generate_multiple_answers import beam_search


def analyze_checkpoints():
    questions, answers = load_data(params.data_file_directory, params.files, None)
    VOCAB_SIZE = 15001

    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, 'UNK')
    # tokenizer = utils.load_and_unpickle("test_models/tokenizer")

    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    reversed_tokenizer_word_dict = {index: text for text, index in tokenizer.word_index.items()}
    mle_model = utils.fit_mle_model(tokenized_answers, reversed_tokenizer_word_dict)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    checkpoints = [params.dir_name + file for file in os.listdir(params.dir_name) if file.endswith("hdf5")]
    print(f"{len(checkpoints)} checkpoints")

    results = defaultdict(list)
    model_score = []

    # model evaluations section
    questions, answers = load_data(params.data_file_directory, params.test_files)
    enc_in_data, dec_in_data, dec_out_data = generate_test_values(questions[:1000], answers[:1000], tokenizer)

    # generating answer and perplexity section
    texts = questions[:5]

    for checkpoint in checkpoints:
        net_model, encoder_inputs, encoder_states, decoder_inputs, \
        decoder_embedding, decoder_lstm, decoder_dense = utils.load_keras_model(checkpoint)

        enc_model, dec_model = conversation.make_inference_models(encoder_inputs, encoder_states, decoder_inputs,
                                                                  decoder_embedding,
                                                                  decoder_lstm, decoder_dense)

        score = net_model.evaluate([enc_in_data, dec_in_data], dec_out_data)
        model_score.append(score)
        print(score)
        for text in texts:
            print(text)
            states_values = enc_model.predict(conversation.str_to_tokens(tokenizer, text, max_len_questions))
            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = tokenizer.word_index['start']
            end_index = tokenizer.word_index['end']

            predictions, _ = beam_search(states_values, empty_target_seq, dec_model, end_index)

            decoded_texts = []
            for prediction in predictions:
                decoded_text = ['start']
                for word_index in prediction[1:]:
                    decoded_text.append(reversed_tokenizer_word_dict.get(word_index, 'UNK'))
                decoded_texts.append(decoded_text)
            result = choose_best_fit(decoded_texts, mle_model)
            results[text].append(result)

    utils.pickle_and_save(results, params.perplexity_file)
    utils.pickle_and_save(model_score, params.model_summary_file)


def generate_test_values(questions, answers, tokenizer):
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)
    prepared_data = prepare_data(tokenized_questions, tokenized_answers)
    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = prepared_data
    return encoder_input_data, decoder_input_data, decoder_output_data


def choose_best_fit(texts, mle_model):
    # bigramer = Bigramer(dict=utils.load_and_unjson("preprocessed_big_train_bigramer"))

    # bigrammed = [bigramer.replace_unks(words) for words in texts]

    without_unks = [words for words in texts if 'UNK' not in words]

    best = None
    if len(without_unks) == 0:
        # jeśli każdy zawiera co najmniej jednego UNK, to wybieramy najlepszy i usuwamy z niego UNKi
        # best = utils.choose_best(bigrammed, mle_model)
        best = utils.choose_best(texts, mle_model, return_score=True)
        # best = bigramer.strip_unks(best)
    else:
        # w przeciwnym razie wybieramy najlepszego spośród tych bez UNKów
        best = utils.choose_best(without_unks, mle_model, return_score=True)

    best = best[0][1:-1], best[1]  # strip 'start' and 'end'
    return best


def show_stats():
    stats = utils.load_and_unpickle(params.perplexity_file)

    for question, results in stats.items():
        print(f"Question: {question}")
        for answer, perplexity in results:
            print(f"{perplexity} -> {answer}")
        print(end="\n\n")

    model_stats = utils.load_and_unpickle(params.model_summary_file)
    for loss, accuracy in model_stats:
        print(loss, accuracy)


if __name__ == '__main__':
    analyze_checkpoints()
    # show_stats()
