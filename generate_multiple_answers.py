import numpy as np
import conversation
import utils
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data

from math import log
import nltk
from nltk.lm import Laplace, Vocabulary

model = Laplace(2)  # bigramy


def fit_mle_model(text, text_dict):
    # text dict key: index value: text, nie ma w tokenizer domyslnie trzeba odwrocic slownik
    tokenized_text = [[text_dict[index] for index in sentence] for sentence in text]
    train_data = [nltk.bigrams(t) for t in tokenized_text]
    words = [word for sentence in tokenized_text for word in sentence]
    vocab = Vocabulary(words)
    model.fit(train_data, vocab)


def calculate_mle(decoded_translations: list) -> list:
    test_data = [nltk.bigrams(t) for t in decoded_translations]
    results = []
    for test in test_data:
        score = 0
        for ngram in test:
            score = score + log(model.score(ngram[-1], ngram[:-1]))
        results.append(score)
    return results


def choose_best(decoded_translations: list):
    # wybiera odpowiedz o najwiekszym prawdopodobienstwie
    results = calculate_mle(decoded_translations)
    # print(results)
    best_index = np.argsort(results)[-1]
    # print(best_index)
    # print(decoded_translations[best_index])
    return decoded_translations[best_index]


def test():
    questions, answers = load_data("prepare_data/output_files", "preprocessed_train")
    VOCAB_SIZE = 15001

    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, 'UNK')
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    reversed_tokenizer_word_dict = {index: text for text, index in tokenizer.word_index.items()}
    fit_mle_model(tokenized_answers, reversed_tokenizer_word_dict)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    _, encoder_inputs, encoder_states, decoder_inputs, \
        decoder_embedding, decoder_lstm, decoder_dense = utils.load_latest_checkpoint()

    enc_model, dec_model = conversation.make_inference_models(encoder_inputs, encoder_states, decoder_inputs,
                                                              decoder_embedding,
                                                              decoder_lstm, decoder_dense)

    texts = ['stop talking shit', 'it is peanut butter jelly time', 'Are we going to pass this lecture',
             'Where are you from',
             'do you like me', 'carrot', 'tell me your biggest secret', 'How are you', 'do you know me',
             'what does fox say', 'i am happy', 'this is america', 'kill me', 'do not forget to brush your teeth']
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
                decoded_text.append(reversed_tokenizer_word_dict[word_index])
            decoded_texts.append(decoded_text)
        print(choose_best(decoded_texts))

        # for prediction in predictions:
        #     decoded_translation = ''
        #     for sampled_word_index in prediction[1:]:
        #         decoded_translation += ' {}'.format(reversed_tokenizer_word_dict[sampled_word_index])
        #     print(decoded_translation)

        # print(predictions)
        print()


def beam_search(states_values, empty_target_seq, dec_model, end_index, k=5, maxsample=60):
    maxsample = maxsample
    eos = end_index
    k = k
    dead_k = 0
    dead_samples = []
    dead_scores = []
    live_scores = [0]
    live_samples = [[['start'], [empty_target_seq] + states_values]]
    live_k = 1

    while live_k and dead_k < k:
        new_live_samples = []
        new_live_scores = []
        for j, live_sample in enumerate(live_samples):
            probs, h, c = dec_model.predict(live_sample[1])
            cand_scores = np.array([live_scores[j]])[:, None] - np.log(probs)
            cand_flat = cand_scores.flatten()

            # find the best (lowest) scores we have from all possible samples and new words
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]
            new_live_scores.extend(cand_flat[ranks_flat])
            # append the new words to their appropriate live sample
            # voc_size = probs.shape[1]
            states_values = [h, c]

            for i, r in enumerate(ranks_flat):
                empty_target_seq = np.zeros((1, 1))
                empty_target_seq[0, 0] = r
                new_live_samples = new_live_samples + [[live_samples[j][0] + [r], [empty_target_seq] + states_values]]

        # choose k-dead best
        live_scores = []
        live_samples = []
        best_so_far = np.array(new_live_scores).argsort()[:(k - dead_k)]
        for best_index in best_so_far:
            live_samples.append(new_live_samples[best_index])
            live_scores.append(new_live_scores[best_index])

        # # live samples that should be dead are...
        zombie = [s[0][-1] == eos or len(s[0]) >= maxsample for s in live_samples]
        #
        # # add zombies to the dead
        dead_samples += [s[0] for s, z in zip(live_samples, zombie) if z]  # remove first label == empty
        dead_scores += [s for s, z in zip(live_scores, zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living
        live_samples = [s for s, z in zip(live_samples, zombie) if not z]
        live_scores = [s for s, z in zip(live_scores, zombie) if not z]
        live_k = len(live_samples)
        #

    return dead_samples, dead_scores


if __name__ == '__main__':
    test()
