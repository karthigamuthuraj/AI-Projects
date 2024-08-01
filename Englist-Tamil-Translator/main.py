from models.preprocess import load_data, preprocess_data
from models.train import train_model
from models.predict import predict_sentence
from embeddings import load_word2vec_model, create_embedding_matrix

def main():
    """
    Main function to load data, preprocess, train the model, and make predictions.
    """
    # Load and preprocess data
    english_sentences, tamil_sentences = load_data('data/engtamilTrain.csv')
    (input_sequences, output_sequences, decoder_input_sequences, decoder_output_sequences,
     english_tokenizer, tamil_tokenizer, english_vocab_size, tamil_vocab_size,
     max_input_seq_length, max_output_seq_length) = preprocess_data(english_sentences, tamil_sentences)

    # Load pre-trained Word2Vec models
    eng_model = load_word2vec_model('embeddings/engmodel.bin')
    tam_model = load_word2vec_model('embeddings/tammodel.bin')

    # Create embedding matrices
    eng_embedding_matrix = create_embedding_matrix(eng_model, english_tokenizer, english_vocab_size)
    tam_embedding_matrix = create_embedding_matrix(tam_model, tamil_tokenizer, tamil_vocab_size)

    # Train the model
    model = train_model(input_sequences, output_sequences, decoder_output_sequences,
                        english_vocab_size, tamil_vocab_size, max_input_seq_length,
                        max_output_seq_length, eng_embedding_matrix, tam_embedding_matrix)

    # Example prediction
    input_sentence = "<SOS> Finally, the columnist fails to tell us who among the political leaders of the bourgeoisie, past and present, he counts among the paragons of morality <EOS>"
    decoded_statement = predict_sentence(model, english_tokenizer, tamil_tokenizer,
                                         input_sentence, max_input_seq_length,
                                         max_output_seq_length)
    print(f"Decoded Statement: {decoded_statement}")

if __name__ == "__main__":
    main()
