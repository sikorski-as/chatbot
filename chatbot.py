import numpy as np
import tensorflow as tf

from tensorflow.python.keras.callbacks import ModelCheckpoint

from conversation import make_inference_models, str_to_tokens
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data

if __name__ == '__main__':
    questions, answers = load_data("prepare_data/output_files", "preprocessed_train")
    # questions = questions[:1000]
    # answers = answers[:1000]
    VOCAB_SIZE = 15001
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, 'UNK')
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_input")
    encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True, name="encoder_embedding")(encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True, name="encoder_lstm")(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_input")
    decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, name="decoder_embedding", mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax, name="decoder_dense")
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    model.summary()

    filepath = "weights.best_test.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)
    callbacks_list = [checkpoint]

    model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list, batch_size=64, epochs=4)
    model.save('model_test.h5')

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

