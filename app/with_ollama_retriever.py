import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_and_create_vectors(pdf_paths):
    """
    Load multiple PDF files and create a vectorstore
    
    Args:
        pdf_paths: Can be a single path (string) or list of paths
    """
    # Ensure pdf_paths is a list
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
    
    # Load documents from all PDFs
    all_documents = []
    
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = os.path.basename(pdf_path)
            
            all_documents.extend(documents)
        else:
            print(f"Warning: PDF file not found: {pdf_path}")
    
    if not all_documents:
        raise ValueError("No valid PDF documents found to process")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    
    print(f"Vectorstore created successfully with {len(chunks)} chunks from {len(pdf_paths)} PDF(s)")
    return vectorstore
