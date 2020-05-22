import os
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from tensorflow_core.python.keras.saving.save import load_model


def create_keras_model(vocab_size):
    encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_input")
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, mask_zero=True, name="encoder_embedding")(
        encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True, name="encoder_lstm")(
        encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_input")
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, name="decoder_embedding", mask_zero=True)(
        decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax, name="decoder_dense")
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense


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


def load_latest_checkpoint():
    VOCAB_SIZE = 15001
    model_info = create_keras_model(VOCAB_SIZE)
    model = model_info[0]

    checkpoint_path = "checkpoints/train1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    model.load_weights(latest)
    return model_info


if __name__ == '__main__':
    load_latest_checkpoint()
