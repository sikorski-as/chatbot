from tensorflow.python.keras.callbacks import ModelCheckpoint
import params
import utils
from data import load_data, create_tokenizer, tokenize_q_a, prepare_data

if __name__ == '__main__':
    questions, answers = load_data(params.data_file_directory, params.files, encoding=None)

    VOCAB_SIZE = params.vocab_size
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, params.unknown_token)
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    prepared_data = prepare_data(tokenized_questions, tokenized_answers)
    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = prepared_data

    created_model = utils.create_keras_model(VOCAB_SIZE)
    model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = created_model

    model.summary()

    checkpoint_path = params.checkpoints_save_path
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 verbose=1,
                                 save_weights_only=params.checkpoints_save_weights_only,
                                 period=params.checkpoints_frequency)
    callbacks_list = [checkpoint]
    # model.save_weights(checkpoint_path.format(epoch=0))

    # model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list,
    #           batch_size=params.batch_size, epochs=params.epochs)
    model.save(params.model_save_path)
