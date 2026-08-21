import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from app.with_ollama_retriever import get_embeddings

load_dotenv()

def load_agent(vector_store_path="vectorstore", model_name="meta-llama/llama-prompt-guard-2-86m", provider="groq", ollama_base_url="http://localhost:11434"):
    """
    Load agent with support for both Groq and Ollama providers
    
    Args:
        vector_store_path: Path to the vector store
        model_name: Name of the model to use
        provider: Either 'groq' or 'ollama'
        ollama_base_url: Base URL for Ollama (only used if provider is 'ollama')
    """
    
    # Load embeddings and vectorstore
    embeddings = get_embeddings()
    db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 1})

    # Initialize LLM based on provider
    if provider.lower() == "groq":
        # Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1
        )
        print(f"✅ Loaded Groq model: {model_name}")
        
    elif provider.lower() == "ollama":
        # Ollama LLM
        try:
            llm = OllamaLLM(
                model=model_name,
                base_url=ollama_base_url,
                temperature=0.1
            )
            print(f"✅ Loaded Ollama model: {model_name} from {ollama_base_url}")
        except Exception as e:
            raise ValueError(f"Failed to connect to Ollama: {str(e)}. Make sure Ollama is running and the model is installed.")
            
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'groq' or 'ollama'")

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )
    
    return qa_chain
