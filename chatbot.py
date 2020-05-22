import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
import conversation
import utils
from conversation import make_inference_models, str_to_tokens
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data

if __name__ == '__main__':
    questions, answers = load_data("prepare_data/output_files", "preprocessed_train")

    VOCAB_SIZE = 15001
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, 'UNK')
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    model, encoder_inputs, encoder_states, \
        decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = utils.create_keras_model(VOCAB_SIZE)

    model.summary()

    checkpoint_path = "checkpoints/train1/cp-{epoch:04d}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1)
    callbacks_list = [checkpoint]
    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list, batch_size=64,
              epochs=2)
    model.save('model_test.h5')
