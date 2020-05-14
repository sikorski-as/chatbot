from tensorflow.keras import layers, activations, models, preprocessing, utils
# import tensorflow as tf
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input
import numpy as np

if __name__ == '__main__':
    answers_file = open("prepare_data/output_files/train2_good.from", encoding="utf-8")
    questions_file = open("prepare_data/output_files/train2_good.to", encoding="utf-8")
    answers_raw = answers_file.readlines()
    questions_raw = questions_file.readlines()
    answers = list()
    for answer in answers_raw:
        answers.append('<START> ' + answer + ' <END>')
    questions = questions_raw

    tokenizer = preprocessing.text.Tokenizer(num_words=15001, oov_token=15000)
    tokenizer.fit_on_texts(questions + answers)
    VOCAB_SIZE = 15001 # len(tokenizer.word_index) + 1
    print('VOCAB SIZE : {}'.format(VOCAB_SIZE))
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])


    model = load_model('model_200_2.h5')

    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]   # input_2
    decoder_embedding = model.layers[3].output
    decoder_lstm = model.layers[5]

    decoder_state_input_h = layers.Input(shape=(200,), name="input_h")
    decoder_state_input_c = layers.Input(shape=(200,), name="input_c")

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)

    def make_inference_models():

        encoder_model = Model(encoder_inputs, encoder_states)


        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def str_to_tokens( sentence : str ):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append( tokenizer.word_index[ word ] )
        return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')

    enc_model, dec_model = make_inference_models()

    for _ in range(10):
        states_values = enc_model.predict(str_to_tokens(input('Enter question : ')))
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

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        print(decoded_translation)