import os
from dotenv import load_dotenv
import qdrant_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant

# --- Configuration ---
# Name for the collection in Qdrant. This will be the database for your PDFs.
QDRANT_COLLECTION_NAME = "indian_legal_judgements"


def get_documents_from_data_folder():
    """
    Loads all PDF documents from the 'data' directory.
    Checks if the folder is empty.
    """
    print("Step 1: Loading documents from the './data' folder...")
    if not os.path.exists('data') or not os.listdir('data'):
        print("\nERROR: The 'data' folder is empty or does not exist.")
        print("Please add your 3 PDF files to the 'data' folder and try again.")
        exit()

    loader = DirectoryLoader(
        'data/',                  # Path to the folder
        glob="**/*.pdf",          # Pattern to match only PDF files
        loader_cls=PyPDFLoader,   # The loader class to use for each file
        show_progress=True,       # Display a progress bar
        use_multithreading=True   # Speed up loading for multiple files
    )
    documents = loader.load()
    print(f"-> Loaded {len(documents)} document(s) successfully.")
    return documents


def split_documents_into_chunks(documents):
    """
    Splits the loaded documents into larger, more contextually relevant chunks.
    This is the most critical step for improving retrieval accuracy.
    """
    print("\nStep 2: Splitting documents into better, context-rich chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,        # Increased size to capture full paragraphs/ideas
        chunk_overlap=200,      # Increased overlap to maintain context between chunks
        add_start_index=True    # Helps in identifying chunk order if needed later
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def main():
    """
    Main function to run the entire data processing and indexing pipeline.
    """
    print("--- Starting Data Ingestion Process ---")

    # Load environment variables from .env file (API keys, etc.)
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Validate that all required environment variables are set
    if not all([google_api_key, qdrant_url, qdrant_api_key]):
        print("\nFATAL ERROR: Required environment variables are not set.")
        print("Please create and configure your .env file with GOOGLE_API_KEY, QDRANT_URL, and QDRANT_API_KEY.")
        return

    # --- Execute Pipeline ---
    # 1. Load the documents from the data folder
    documents = get_documents_from_data_folder()

    # 2. Split the documents into smaller chunks
    texts = split_documents_into_chunks(documents)

    # 3. Initialize the embedding model from Google
    print("\nStep 3: Initializing Google's Gemini embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    print("-> Embedding model initialized.")

    # 4. Create or connect to the Qdrant vector store and upload the documents
    # This is the final step where the magic happens.
    # We use `force_recreate=True` to ensure we start fresh every time.
    print(f"\nStep 4: Uploading {len(texts)} chunks to Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    print("(This will automatically delete any existing collection with the same name)")
    
    Qdrant.from_documents(
        texts,
        embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=QDRANT_COLLECTION_NAME,
        prefer_grpc=True,
        force_recreate=True,  # Automatically deletes old data. Perfect for development.
    )

    print("\n" + "-"*50)
    print("✅ INGESTION COMPLETE!")
    print(f"Your {len(documents)} PDF(s) have been successfully processed and indexed.")
    print("You can now run 'streamlit run app.py' to ask questions.")
    print("-" * 50)


if __name__ == "__main__":
    main()