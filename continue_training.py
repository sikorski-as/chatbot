from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model

from data import load_data, create_tokenizer, tokenize_q_a, prepare_data
from utils import load_latest_checkpoint

if __name__ == '__main__':
    questions, answers = load_data("prepare_data/output_files", "preprocessed_train")
    VOCAB_SIZE = 15001
    tokenizer = create_tokenizer(questions + answers, VOCAB_SIZE, 'UNK')
    tokenized_questions, tokenized_answers = tokenize_q_a(tokenizer, questions, answers)

    max_len_questions, max_len_answers, encoder_input_data, decoder_input_data, decoder_output_data = \
        prepare_data(tokenized_questions, tokenized_answers)

    # new_model = load_model('model_test.h5')
    new_model = load_latest_checkpoint()[0]

    checkpoint_path = "checkpoints/train2/cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1)
    callbacks_list = [checkpoint]

    # fit the model
    new_model.fit([encoder_input_data, decoder_input_data], decoder_output_data, callbacks=callbacks_list, batch_size=128,
              epochs=4)
    # new_model.save('model_test.h5')