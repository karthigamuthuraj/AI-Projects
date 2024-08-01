import numpy as np
from gensim.models import Word2Vec

def load_word2vec_model(path):
    """
    Load a pre-trained Word2Vec model.
    
    Args:
    path (str): Path to the Word2Vec model file.

    Returns:
    Word2Vec: Loaded Word2Vec model.
    """
    return Word2Vec.load(path)

def create_embedding_matrix(word2vec_model, tokenizer, vocab_size):
    """
    Create an embedding matrix from a Word2Vec model.
    
    Args:
    word2vec_model (Word2Vec): Pre-trained Word2Vec model.
    tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer for the text.
    vocab_size (int): Size of the vocabulary.

    Returns:
    np.ndarray: Embedding matrix.
    """
    embedding_matrix = np.zeros((vocab_size, word2vec_model.vector_size))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    return embedding_matrix
