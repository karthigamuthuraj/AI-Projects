import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from embeddings import create_embedding_matrix

def create_seq2seq_model(input_vocab_size, output_vocab_size, input_seq_length, output_seq_length, hidden_units, eng_embedding_matrix, tam_embedding_matrix):
    """
    Build a Seq2Seq model with LSTM layers for translation.
    
    Args:
    input_vocab_size (int): Size of the English vocabulary.
    output_vocab_size (int): Size of the Tamil vocabulary.
    input_seq_length (int): Maximum length of input sequences.
    output_seq_length (int): Maximum length of output sequences.
    hidden_units (int): Number of hidden units in LSTM layers.
    eng_embedding_matrix (np.ndarray): Pre-trained English word embeddings.
    tam_embedding_matrix (np.ndarray): Pre-trained Tamil word embeddings.

    Returns:
    model (tf.keras.Model): Compiled Seq2Seq model.
    """
    # Encoder
    encoder_inputs = Input(shape=(input_seq_length,))
    encoder_embedding = Embedding(input_vocab_size, hidden_units, weights=[eng_embedding_matrix], trainable=False)(encoder_inputs)
    encoder_lstm, encoder_state_h, encoder_state_c = LSTM(hidden_units, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(output_seq_length,))
    decoder_embedding = Embedding(output_vocab_size, hidden_units, weights=[tam_embedding_matrix], trainable=False)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[encoder_state_h, encoder_state_c])
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model(input_sequences, output_sequences, decoder_output_sequences, english_vocab_size, tamil_vocab_size, max_input_seq_length, max_output_seq_length, eng_embedding_matrix, tam_embedding_matrix):
    """
    Train the Seq2Seq model on the provided data.
    
    Args:
    input_sequences (np.ndarray): Padded sequences of English sentences.
    output_sequences (np.ndarray): Padded sequences of Tamil sentences.
    decoder_output_sequences (np.ndarray): One-hot encoded target sequences for the decoder.
    english_vocab_size (int): Size of the English vocabulary.
    tamil_vocab_size (int): Size of the Tamil vocabulary.
    max_input_seq_length (int): Maximum length of input sequences.
    max_output_seq_length (int): Maximum length of output sequences.
    eng_embedding_matrix (np.ndarray): Pre-trained English word embeddings.
    tam_embedding_matrix (np.ndarray): Pre-trained Tamil word embeddings.

    Returns:
    model (tf.keras.Model): Trained Seq2Seq model.
    """
    model = create_seq2seq_model(english_vocab_size, tamil_vocab_size, max_input_seq_length, max_output_seq_length, 100, eng_embedding_matrix, tam_embedding_matrix)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 32
    epochs = 500
    model.fit([input_sequences, output_sequences], decoder_output_sequences, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model
