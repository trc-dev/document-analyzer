import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from app.retriever import get_embeddings

load_dotenv()

def load_agent(vector_store_path="vectorstore", model_name="llama-3.3-70b-versatile"):
    """
    Load agent with enhanced error handling and deployment compatibility
    
    Args:
        vector_store_path: Path to the vector store
        model_name: Name of the Groq model to use
    """
    
    try:
        # Load embeddings and vectorstore
        print(f"🔍 Loading embeddings and vectorstore from: {vector_store_path}")
        embeddings = get_embeddings()
        
        if not os.path.exists(vector_store_path):
            raise ValueError(f"Vector store not found at: {vector_store_path}")
        
        db = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector store loaded successfully")
        
        # Get API key - try Streamlit secrets first, then environment
        groq_api_key = None
        try:
            # For Streamlit Cloud deployment
            groq_api_key = st.secrets["GROQ_API_KEY"]
            print("✅ API key loaded from Streamlit secrets")
        except:
            # For local development
            groq_api_key = os.getenv("GROQ_API_KEY")
            print("✅ API key loaded from environment variables")
        
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please add it to Streamlit secrets or environment variables.\n"
                "Get your API key from: https://console.groq.com/"
            )
        
        # Initialize Groq LLM
        print(f"🔍 Initializing Groq model: {model_name}")
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1000
        )
        print("✅ Groq model initialized successfully")
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        print("✅ QA chain created successfully")
        return qa_chain
        
    except Exception as e:
        print(f"❌ Error in load_agent: {str(e)}")
        raise e
