from keras_preprocessing.text import Tokenizer

if __name__ == '__main__':

    num_words = 3
    tk = Tokenizer(oov_token='UNK', num_words=num_words+1)
    texts = ["my name is far faraway asdasd", "my name is","your name is"]
    tk.fit_on_texts(texts)
    print(tk.word_index)
    print(tk.texts_to_sequences(texts))
    ## **Key Step**
    tk.word_index = {e:i for e,i in tk.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
    tk.word_index[tk.oov_token] = num_words + 1
    print(tk.word_index)
    print(tk.texts_to_sequences(texts))