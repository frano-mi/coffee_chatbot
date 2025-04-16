import os
import time
from typing import List, Dict
import streamlit as st
import requests
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ========== CONFIGURATION ==========
class AppConfig:
    """Centralized configuration for the application"""
    PAGE_TITLE = "☕ Brazilian Coffee Assistant"
    PAGE_ICON = "☕"
    KNOWLEDGE_BASE_DIR = "./knowledge_base"
    CHROMA_DB_PATH = "./chroma_db" # Path where Chroma vector database will be stored
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    OLLAMA_MODEL = "llama3.1"
    OLLAMA_API_URL = "http://localhost:11434/api/chat"
    EXAMPLE_QUESTIONS = [
        "What's special about your Brazilian coffee?",
        "Which coffee products do you offer?",
        "How is your coffee shipped?",
        "Is your coffee fair-trade certified?"
    ]

# ========== UTILITIES ==========
def load_documents(folder_path: str) -> List[str]:
    """Load and validate documents from knowledge base directory"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Knowledge base directory not found at {folder_path}")
    
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".md", ".txt")):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip() # Read content and remove extra spaces
                    if content:  # Skip empty files
                        documents.append(content) # Add content to the documents list
            except UnicodeDecodeError:
                st.warning(f"Skipped {filename} - invalid encoding")
    return documents

# ========== AI COMPONENTS ==========
class CoffeeAI:
    """Encapsulates all AI-related functionality"""
    
    def __init__(self):
        self.embeddings = self._load_embedding_model()
        self.vector_db = self._initialize_vector_db(self.embeddings)
    
    @st.cache_resource(show_spinner="Loading embedding model...")
    def _load_embedding_model(_self):
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    @st.cache_resource(show_spinner="Indexing knowledge base...")
    def _initialize_vector_db(_self, _embeddings):
        documents = load_documents(AppConfig.KNOWLEDGE_BASE_DIR) # Load documents from knowledge base
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n## ", "\n\n", "\n", ".", " ", ""]
        )
        
        all_chunks = []
        for doc in documents:
            chunks = splitter.split_text(doc)  # Split document into chunks
            all_chunks.extend(chunks)
        
        return Chroma.from_texts(
            texts=all_chunks,
            embedding=_embeddings,
            persist_directory=AppConfig.CHROMA_DB_PATH
        )
    
    def get_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from knowledge base"""
        docs = self.vector_db.similarity_search(query, k=k) # Retrieve top-k relevant documents
        return "\n".join(doc.page_content for doc in docs)

class ChatManager:
    """Handles all chat-related operations"""
    
    @staticmethod
    def format_prompt(context: str, question: str) -> List[Dict]:
        """Structure the LLM prompt with context and question"""
        return [
            {
                "role": "system", # System message to instruct the model
                "content": (
                    "You're a knowledgeable Brazilian coffee expert. Provide concise, "
                    "accurate answers about our premium coffee products. "
                    "Be friendly and professional. If you don't know an answer, "
                    "say you'll check with a human expert."
                )
            },
            {
                "role": "user", # User's question and context
                "content": (
                    f"Here's some relevant information:\n{context}\n\n"
                    f"Based on this, please answer: {question}\n"
                    "Keep response under 3 sentences unless more detail is needed."
                )
            }
        ]
    
    @staticmethod
    def query_llm(messages: List[Dict], timeout: int = 90, retries: int = 2) -> str:
        """Send query to LLM with enhanced error handling"""
        last_exception = None
        
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    AppConfig.OLLAMA_API_URL, # Send request to Ollama API
                    json={
                        "model": AppConfig.OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()["message"]["content"] # Return the content of the response
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < retries:
                    time.sleep((attempt + 1) * 2)  # Exponential backoff
                continue
        
        return (
            "⚠️ Our coffee experts are currently unavailable.\n\n"
            f"Technical details: {str(last_exception)[:200]}..."
        )

# ========== UI COMPONENTS ==========
def initialize_chat():
    """Initialize chat session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_query_time = 0

def render_chat_history():
    """Display chat messages with avatars"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="☕" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

def render_example_questions():
    """Display clickable example questions"""
    st.subheader("Need inspiration?")
    cols = st.columns(2)
    for i, question in enumerate(AppConfig.EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"example_{hash(question)}"): # Add buttons for example questions
                return question
    return None

def render_sidebar():
    """Additional UI elements in sidebar"""
    with st.sidebar:
        st.header("About")
        st.markdown("""
            This assistant specializes in:
            - Brazilian coffee varieties
            - Shipping & logistics
            - Wholesale inquiries
            - Brewing recommendations
            """)
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ========== MAIN APP ==========
def main():
    # App setup
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide"
    )
    os.environ["STREAMLIT_WATCH_DISABLE"] = "true" # Disable Streamlit watch
    
    # Initialize components
    initialize_chat()
    ai = CoffeeAI()
    
    # UI Layout
    st.title(AppConfig.PAGE_TITLE)
    st.caption("Your expert guide to premium Brazilian coffee")
    
    col1, col2 = st.columns([3, 1])

    with col1:
        render_chat_history()
        
        # Handle example questions first
        example_question = render_example_questions()
        
        # Get user input (either from chat or example buttons)
        user_input = st.chat_input("Ask about our coffee...")
        user_input = example_question or user_input
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Force a rerun to display the user message first
            st.rerun()

        # Check if the last message is from the user (meaning we need to generate a response)
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Brewing your response..."):
                try:
                    context = ai.get_context(st.session_state.messages[-1]["content"])
                    messages = ChatManager.format_prompt(context, st.session_state.messages[-1]["content"])
                    response = ChatManager.query_llm(messages)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Force a rerun to display the assistant message
                    st.rerun()
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")   
                     
    with col2:
        render_sidebar()

if __name__ == "__main__":
    main() # Start the main function when the script runs