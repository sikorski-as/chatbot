#
# for reddit dataset
#

# overall
encoding = 'utf8'
vocab_size = 15001
unknown_token = "UNK"

# data preparation
data_file_directory = "prepare_data/output_files"
files = "preprocessed_big_train"

# model creation
model_save_path = "model1.h5"
encoder_embedding_size = 200
encoder_lstm_size = 200
decoder_lstm_size = 200
decoder_state_input_h_size = 200
decoder_state_input_c_size = 200

# training
epochs = 1
batch_size = 128
checkpoints_frequency = 5
checkpoints_save_weights_only = False
checkpoints_save_path = "checkpoints/train1/cp-{epoch:04d}.hdf5"

# usage
bigramer = "preprocessed_big_train_bigramer"
model = "checkpoints/train2/cp-0030.hdf5"

#
# for cornell dataset
#

# # overall
# encoding = None
# vocab_size = 15001
# unknown_token = "UNK"
#
# # data preparation
# data_file_directory = "prepare_data/output_files"
# files = "preprocessed_cornell"
#
# # model creation
# model_save_path = "model1.h5"
# encoder_embedding_size = 200
# encoder_lstm_size = 200
# decoder_lstm_size = 200
# decoder_state_input_h_size = 200
# decoder_state_input_c_size = 200
#
# # training
# epochs = 1
# batch_size = 128
# checkpoints_frequency = 5
# checkpoints_save_weights_only = False
# checkpoints_save_path = "checkpoints/train1/cp-{epoch:04d}.hdf5"
#
# # usage
# bigramer = "preprocessed_cornell_bigramer"
# model = "checkpoints/cornell/cp-0090.hdf5"