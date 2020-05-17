from tensorflow_core.python.keras.saving.save import load_model


def load_keras_model(file):
    model = load_model(file)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]

    decoder_inputs = model.input[1]  # input_2
    decoder_embedding = model.layers[3].output
    decoder_lstm = model.layers[5]
    decoder_dense = model.layers[6]

    return encoder_inputs, encoder_states, decoder_inputs, \
           decoder_embedding, decoder_lstm, decoder_dense
