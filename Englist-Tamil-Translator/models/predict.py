import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentence(model, english_tokenizer, tamil_tokenizer, input_sentence, max_input_seq_length, max_output_seq_length):
    """
    Predict the translation of an input sentence using the trained model.
    
    Args:
    model (tf.keras.Model): Trained Seq2Seq model.
    english_tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer for English sentences.
    tamil_tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer for Tamil sentences.
    input_sentence (str): Sentence in English to be translated.
    max_input_seq_length (int): Maximum length of input sequences.
    max_output_seq_length (int): Maximum length of output sequences.

    Returns:
    str: Translated Tamil sentence.
    """
    # Convert input sentence to sequence and pad it
    input_sequence = english_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=max_input_seq_length, padding='post')

    # Generate predictions
    predictions = model.predict([input_sequence, np.zeros((1, max_output_seq_length))])
    predicted_tokens = np.argmax(predictions, axis=-1)[0]

    # Convert tokens to words
    tamil_index_word = {i: w for w, i in tamil_tokenizer.word_index.items()}
    decoded_sentence = [tamil_index_word.get(token, '<unk>') for token in predicted_tokens if token != 0 and token != tamil_tokenizer.word_index.get('<EOS>', 0)]

    decoded_statement = ' '.join(decoded_sentence)
    return decoded_statement
