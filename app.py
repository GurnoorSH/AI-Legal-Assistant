import os
import streamlit as st
from dotenv import load_dotenv
import qdrant_client
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant

# Add these two lines to solve the event loop error on Windows
import nest_asyncio
nest_asyncio.apply()

# --- Configuration ---
QDRANT_COLLECTION_NAME = "indian_legal_judgements"


# --- Backend Logic (Cached) ---

@st.cache_resource
def get_qdrant_retriever(_qdrant_url, _qdrant_api_key, _google_api_key):
    """Initializes and caches the Qdrant retriever."""
    print("Initializing Qdrant client and retriever...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_google_api_key)
    client = qdrant_client.QdrantClient(url=_qdrant_url, api_key=_qdrant_api_key)
    vector_store = Qdrant(client=client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 5}) # Increased k to 5 for more comprehensive context

@st.cache_resource
def get_rag_chain(_llm, _retriever):
    """Initializes and caches the RAG chain."""
    print("Initializing RAG chain...")
    prompt_template = """
    You are a professional legal research assistant AI. Your purpose is to provide precise, factual, and objective answers based *only* on the legal document excerpts provided to you as context.

    GUIDELINES:
    - Synthesize a comprehensive answer from the provided context. Do not add any information that is not explicitly present in the text.
    - If the context does not contain the answer, you must state: "Based on the provided documents, I cannot find an answer to this question."
    - After your answer, you must cite the sources you used. List each source on a new line.

    CONTEXT:
    {context}

    QUESTION:
    {input}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    question_answer_chain = create_stuff_documents_chain(_llm, prompt)
    return create_retrieval_chain(_retriever, question_answer_chain)


# --- Main Application UI and Logic ---

def main():
    """The main function to run the Streamlit application."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([google_api_key, qdrant_url, qdrant_api_key]):
        st.error("🚨 Critical Error: Required API keys or URL are missing.")
        st.info("Please ensure your .env file is correctly configured with GOOGLE_API_KEY, QDRANT_URL, and QDRANT_API_KEY.")
        return

    # --- Page Configuration (MUST be the first Streamlit command) ---
    st.set_page_config(
        page_title="AI Legal Research Assistant",
        page_icon="⚖️",
        layout="wide"
    )
    
    # --- Sidebar ---
    with st.sidebar:
        st.image("assets/emblem.png")
        st.title("About this App")
        st.info(
            "This AI-powered application allows legal professionals to ask natural language questions "
            "about an internal database of Indian legal judgments." \
            "   "
            "Made by [Gurnoor Singh]"
        )
        
        st.subheader("How to Use")
        st.markdown(
            """
            1.  Enter a clear question in the text box.
            2.  Click 'Search' to get a sourced answer.
            3.  Review the AI-generated answer and expand the 'Sources' section to see the exact text used.
            """
        )
        
        st.subheader("Example Questions")
        st.markdown("- *What are the key arguments regarding writ petitions in [case name]?*")
        st.markdown("- *WHAT IS  CONTRACTUAL OUSTER OF THE NORMAL JUDICIAL PROCESS ?")
        st.markdown("- *CAN WORDS BE READ INTO SECTION 34? *")
        st.divider()
        st.caption("Powered by Google Gemini & Qdrant")

    # --- Main Content Area ---
    st.title("⚖️ AI Legal Research Assistant")

    # Using a form for a more professional input experience
    with st.form("search_form"):
        user_question = st.text_input(
            "**Enter your legal query below:**",
            placeholder="e.g., Summarize key arguments from cases involving patent infringement...",
            help="Type your question and press 'Search'."
        )
        submit_button = st.form_submit_button("Search")

    if submit_button and user_question:
        try:
            with st.spinner("Analyzing documents... Please wait."):
                # Initialize LLM, retriever, and RAG chain (these will be cached)
                llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.1)
                retriever = get_qdrant_retriever(qdrant_url, qdrant_api_key, google_api_key)
                rag_chain = get_rag_chain(llm, retriever)
                
                # Invoke the chain
                response = rag_chain.invoke({"input": user_question})
                st.session_state.response = response # Store response in session state
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.info("Please try rephrasing your question or check the API key configurations.")

    # Display results if they exist in the session state
    if "response" in st.session_state and st.session_state.response:
        response = st.session_state.response
        
        st.divider()
        st.subheader("Analysis Results")
        
        # Display the answer in a visually distinct container
        with st.container(border=True):
            st.write(response['answer'])

        # Use an expander for the sources to keep the UI clean
        with st.expander("**Click to view sources**"):
            st.subheader("Sources Used for This Answer")
            if 'context' in response and response['context']:
                for doc in response['context']:
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown Source'))
                    page_number = doc.metadata.get('page', 'N/A')
                    
                    with st.container(border=True):
                        st.markdown(f"**📄 Source:** `{source_file}` | **Page:** `{page_number}`")
                        st.markdown(f"**Content:**\n\n> {doc.page_content.strip()}")
            else:
                st.warning("No context documents were returned for this query.")


if __name__ == "__main__":
    main()