import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    """
    Load and preprocess the dataset.
    
    Args:
    file_path (str): Path to the CSV file containing English and Tamil sentences.

    Returns:
    tuple: Contains English and Tamil sentence Series.
    """
    train = pd.read_csv(file_path).drop(["Unnamed: 0"], axis=1)
    english_sentences = train["en"].head(1000)
    tamil_sentences = train["ta"].head(1000)
    return english_sentences, tamil_sentences

def add_sos_eos(series_sentence):
    """
    Add <SOS> and <EOS> tokens to each sentence in the series.
    
    Args:
    series_sentence (pd.Series): Series of sentences to be processed.

    Returns:
    list: List of sentences with <SOS> and <EOS> tokens.
    """
    sos_token, eos_token = "<SOS>", "<EOS>"
    return [f"{sos_token} {sentence} {eos_token}" for sentence in series_sentence]

def preprocess_data(english_sentences, tamil_sentences):
    """
    Preprocess English and Tamil sentences for Seq2Seq model.
    
    Args:
    english_sentences (pd.Series): Series of English sentences.
    tamil_sentences (pd.Series): Series of Tamil sentences.

    Returns:
    tuple: Contains processed sequences, tokenizers, vocab sizes, and max sequence lengths.
    """
    english_sent_SE = add_sos_eos(english_sentences)
    tamil_sent_SE = add_sos_eos(tamil_sentences)

    # Tokenization and padding for English sentences
    english_tokenizer = Tokenizer(filters="")
    english_tokenizer.fit_on_texts(english_sent_SE)
    english_vocab_size = len(english_tokenizer.word_index) + 1
    english_sequences = english_tokenizer.texts_to_sequences(english_sent_SE)

    # Tokenization and padding for Tamil sentences
    tamil_tokenizer = Tokenizer(filters="")
    tamil_tokenizer.fit_on_texts(tamil_sent_SE)
    tamil_vocab_size = len(tamil_tokenizer.word_index) + 1
    tamil_sequences = tamil_tokenizer.texts_to_sequences(tamil_sent_SE)

    max_input_seq_length = 20
    max_output_seq_length = 20

    input_sequences = pad_sequences(english_sequences, maxlen=max_input_seq_length, padding='post')
    output_sequences = pad_sequences(tamil_sequences, maxlen=max_output_seq_length, padding='post')

    # Prepare decoder inputs and outputs
    decoder_input_sequences = np.zeros_like(output_sequences)
    decoder_input_sequences[:, 1:] = output_sequences[:, :-1]
    decoder_input_sequences[:, 0] = tamil_tokenizer.word_index.get('<SOS>', 0)

    decoder_output_sequences = np.eye(tamil_vocab_size)[output_sequences]

    return (input_sequences, output_sequences, decoder_input_sequences, decoder_output_sequences,
            english_tokenizer, tamil_tokenizer, english_vocab_size, tamil_vocab_size,
            max_input_seq_length, max_output_seq_length)
