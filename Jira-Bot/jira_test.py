import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set environment variable to handle OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load and preprocess PDF
pdf_path = 'jira_issues.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Text splitting configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
documents = text_splitter.split_documents(docs)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Create FAISS vector store
try:
    faiss_db = FAISS.from_documents(documents, embeddings)
    print("FAISS vector store created successfully.")
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")

# Function to generate a response from the language model
def generate_response(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Set pad_token_id to eos_token_id if pad_token_id is not set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=pad_token_id
    )
    
    # Decode the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Define and process query
query = "Summary of Issue Key KAN-1"

try:
    # Perform similarity search with the query
    faiss_results = faiss_db.similarity_search(query)
    
    if faiss_results:
        # Get the first result and process it
        first_result = faiss_results[0]
        first_result_text = (first_result['page_content'] 
                             if isinstance(first_result, dict) and 'page_content' in first_result 
                             else str(first_result))
        
        print("FAISS Results:")
        print(first_result_text)
        
        # Generate a response using the language model
        response_prompt = (f"Based on the following text, provide a detailed paragraph summarizing the issue:\n\n"
                           f"{first_result_text}\n\n"
                           "Write a clear and concise summary that encapsulates the main points and findings of the issue.")
        response = generate_response(response_prompt)
        
        print("Generated Response:")
        print(response)
    else:
        print("FAISS similarity search returned no results.")
except Exception as e:
    print(f"Error during FAISS similarity search or response generation: {e}")
