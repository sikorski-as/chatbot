from tensorflow.python.keras.callbacks import ModelCheckpoint

import params
import utils
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data


if __name__ == '__main__':
    questions, answers = load_data(params.data_file_directory, params.files)
    VOCAB_SIZE = params.vocab_size
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, params.unknown_token)
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    # new_model = load_model('model_test.h5')
    new_model = utils.load_keras_model('checkpoints/train2/cp-0004.hdf5')[0]

    """
        Może nadpisać poprzednie checkpointy!!!!!!!! nie zacznie od checkpoint + 1 tylko od 1
    """
    checkpoint_path = params.checkpoints_save_path
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=params.checkpoints_save_weights_only, period=params.checkpoints_frequency)
    callbacks_list = [checkpoint]

    # fit the model
    new_model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list, batch_size=params.batch_size,
              epochs=params.epochs)
