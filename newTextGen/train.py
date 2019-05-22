from textgenrnn import textgenrnn

textgen = textgenrnn(name="new_model")
textgen.reset()
textgen.train_from_file('trumptweet3.txt',
                        new_model=True,
                        rnn_bidirectional=True,
                        rnn_size=64,
                        dim_embeddings=300,
                        num_epochs=50)

print(textgen.model.summary())