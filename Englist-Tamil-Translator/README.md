# Translation Model

## Notes

- **Training and Testing Limitations**: The training and testing are conducted on a limited dataset due to the substantial computational resources required (large CPU and GPU). This project serves as a demonstration and proof of concept rather than a fully optimized and scalable solution.

## Project Overview

The Translation Model is a machine translation system designed to translate English sentences into Tamil using a Sequence-to-Sequence (Seq2Seq) architecture with Long Short-Term Memory (LSTM) networks. This project utilizes pre-trained Word2Vec embeddings to enhance translation quality. The model is built to handle a variety of English sentences and produce Tamil translations, making it a versatile tool for language translation tasks.

## Key Features

- **Sequence-to-Sequence Model**: Utilizes LSTM-based Seq2Seq architecture for translating sentences.
- **Pre-trained Word2Vec Embeddings**: Leverages Word2Vec embeddings for enhanced representation of words.
- **Tokenization and Padding**: Implements tokenization and padding for input and output sequences to ensure consistent model input.
- **Teacher Forcing**: Uses teacher forcing during training to improve the model's performance and convergence.
- **Modular Design**: Organized into modular components for preprocessing, model training, and prediction.

## Technologies Used

- **Python**: Programming language used for implementing the project.
- **TensorFlow/Keras**: Frameworks for building and training the Seq2Seq model.
- **Pandas**: Library for data manipulation and analysis.
- **NumPy**: Library for numerical operations.
- **Gensim**: Library for Word2Vec model loading and manipulation.

## Project Structure

- **`data/engtamilTrain.csv`**: Dataset containing English and Tamil sentence pairs.
- **`models/preprocess.py`**: Contains functions for loading data, preprocessing text, and creating tokenizers.
- **`models/train.py`**: Contains functions to create and train the Seq2Seq model.
- **`models/predict.py`**: Contains functions for making predictions using the trained model.
- **`embeddings/__init__.py`**: Contains functions for loading Word2Vec models and creating embedding matrices.
- **`main.py`**: Main script that ties everything together and executes the training and prediction pipeline.
- **`requirements.txt`**: List of dependencies required for the project.

## Requirements

Ensure you have the necessary Python packages installed. Use the `requirements.txt` file to install them:

```bash
pip install -r requirements.txt
```

## Flowchart

The following flowchart describes the overall process of the Seq2Seq Translation Model: ![image](images/flo.png)


1. **Start**
   - Begin the process.

2. **Load Data**
   - **Action**: Load dataset (`engtamilTrain.csv`).
   - **Output**: English and Tamil sentences.

3. **Preprocess Data**
   - **Action**: 
     - Add `<SOS>` and `<EOS>` tokens.
     - Tokenize and pad sequences.
   - **Output**: Tokenized and padded sequences.

4. **Load Word2Vec Models**
   - **Action**: Load pre-trained Word2Vec models for English and Tamil.
   - **Output**: Word2Vec models.

5. **Create Embedding Matrices**
   - **Action**: Create embedding matrices using the Word2Vec models and tokenizers.
   - **Output**: Embedding matrices.

6. **Build Seq2Seq Model**
   - **Action**: Define encoder and decoder, create LSTM layers, integrate embeddings.
   - **Output**: Seq2Seq model.

7. **Train Model**
   - **Action**: Train the Seq2Seq model with the sequences.
   - **Output**: Trained model.

8. **Predict Sentence**
   - **Action**: Convert input sentence to sequence, predict translation, decode to text.
   - **Output**: Translated Tamil sentence.

9. **End**
   - Conclude the process.

## Usage

1. **Prepare the Data**

   Place your dataset in the `data` folder. Ensure it is named `engtamilTrain.csv` and contains columns `en` (English sentences) and `ta` (Tamil sentences).

2. **Load Pre-trained Word2Vec Models**

   Ensure you have pre-trained Word2Vec models saved as `engmodel.bin` and `tammodel.bin` in the `embeddings` directory.

3. **Run the Main Script**

   Execute `main.py` to run the entire pipeline. This script will:
   - Load and preprocess the data.
   - Load pre-trained Word2Vec models.
   - Create and train the Seq2Seq model.
   - Make predictions on a sample input sentence.

   Run the script using:

   ```bash
   python main.py
   ```

4. **Inspect Results**

   The script will output the translated Tamil sentence for a sample English input. You can modify the `input_sentence` in `main.py` to test different sentences.

## Code Structure

### `models/preprocess.py`

- **`load_data(file_path)`**: Loads data from a CSV file and returns English and Tamil sentences.
- **`add_sos_eos(series_sentence)`**: Adds start-of-sequence (<SOS>) and end-of-sequence (<EOS>) tokens to each sentence.
- **`preprocess_data(english_sentences, tamil_sentences)`**: Tokenizes and pads sentences, prepares sequences for the Seq2Seq model.

### `models/train.py`

- **`create_seq2seq_model(input_vocab_size, output_vocab_size, input_seq_length, output_seq_length, hidden_units, eng_embedding_matrix, tam_embedding_matrix)`**: Builds a Seq2Seq model with LSTM layers.
- **`train_model(input_sequences, output_sequences, decoder_output_sequences, english_vocab_size, tamil_vocab_size, max_input_seq_length, max_output_seq_length, eng_embedding_matrix, tam_embedding_matrix)`**: Trains the Seq2Seq model on the provided data.

### `models/predict.py`

- **`predict_sentence(model, english_tokenizer, tamil_tokenizer, input_sentence, max_input_seq_length, max_output_seq_length)`**: Predicts the Tamil translation of an English input sentence.

### `embeddings/__init__.py`

- **`load_word2vec_model(path)`**: Loads a Word2Vec model from a file.
- **`create_embedding_matrix(word2vec_model, tokenizer, vocab_size)`**: Creates an embedding matrix using a Word2Vec model and a tokenizer.

### `main.py`

- Main script that orchestrates data loading, preprocessing, model training, and prediction.

## Troubleshooting

- **Data Issues**: Ensure your dataset has the correct format and columns.
- **Model Files**: Verify the paths to the pre-trained Word2Vec models.
- **Dependencies**: Install required packages using `requirements.txt`.

