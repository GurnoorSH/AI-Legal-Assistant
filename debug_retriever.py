import os
from dotenv import load_dotenv
import qdrant_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant

# --- Configuration ---
QDRANT_COLLECTION_NAME = "indian_legal_judgements"

def main():
    """
    This script tests the retriever to see what documents it fetches for a given query.
    """
    # Load environment variables from the .env file
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([google_api_key, qdrant_url, qdrant_api_key]):
        print("🚨 Error: Missing required environment variables.")
        return

    # --- Use the exact same components as your main app ---
    print("Initializing embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    print("Connecting to Qdrant...")
    client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    vector_store = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embeddings,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # --- The Test ---
    # !!! IMPORTANT: CHANGE THIS QUESTION TO BE VERY SPECIFIC !!!
    # Use keywords and names that you KNOW are inside your PDF documents.
    test_question = "Numaligarh Refinery Ltd. v. Daelim Industrial Co. Ltd. - What was the Supreme Court's decision regarding the arbitration clause in this case?"

    print(f"\nTesting retriever with question: '{test_question}'")
    print("-" * 50)

    # Invoke the retriever to get the relevant documents
    retrieved_docs = retriever.invoke(test_question)

    if not retrieved_docs:
        print("\n❌ Retrieval failed. No documents were returned.")
    else:
        print(f"\n✅ Retrieval successful. Found {len(retrieved_docs)} documents:\n")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
            print("\nContent:")
            print(doc.page_content)
            print("-" * 50)

if __name__ == "__main__":
    main()