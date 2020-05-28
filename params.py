data_file_directory = "prepare_data/output_files"

files = "preprocessed_cornell"

bigramer = "bigramer_best.json"

encoding = None

vocab_size = 15001

batch_size = 128

epochs = 1

model = "cornell.hdf5"

checkpoints_save_path = "checkpoints/train1/cp-{epoch:04d}.hdf5"
model_save_path = "model1.h5"
checkpoints_frequency = 5
checkpoints_save_weights_only = False

unknown_token = "UNK"

encoder_embedding_size = 200
encoder_lstm_size = 200
decoder_lstm_size = 200

decoder_state_input_h_size = 200
decoder_state_input_c_size = 200
