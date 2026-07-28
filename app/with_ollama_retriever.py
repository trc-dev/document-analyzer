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
            
            try:
                documents = loader.load()
                total_chars = sum(len(doc.page_content.strip()) for doc in documents) if documents else 0
                
                if total_chars < 50:
                    print(f"⚠️ Very little text extracted ({total_chars} chars). PDF might be scanned. Triggering OCR fallback...")
                    raise Exception("Scanned PDF detected")
            except Exception as load_error:
                print(f"❌ Initial load failed or scanned PDF detected for {pdf_path}: {load_error}")
                
                try:
                    print(f"🔄 Attempting OCR processing for: {pdf_path}")
                    from pdf2image import convert_from_path
                    import pytesseract
                    from langchain_core.documents import Document
                    
                    images = convert_from_path(pdf_path)
                    text_content = ""
                    
                    for i, image in enumerate(images):
                        try:
                            page_text = pytesseract.image_to_string(image)
                            text_content += page_text + "\n"
                        except Exception as page_error:
                            print(f"⚠️ Error running OCR on page {i}: {page_error}")
                            continue
                            
                    if text_content.strip():
                        documents = [Document(
                            page_content=text_content,
                            metadata={
                                'source': pdf_path,
                                'source_file': os.path.basename(pdf_path)
                            }
                        )]
                        print(f"✅ OCR processing succeeded for: {pdf_path}")
                    else:
                        documents = []
                        print(f"❌ No text content extracted using OCR from: {pdf_path}")
                except Exception as ocr_error:
                    documents = []
                    print(f"❌ OCR processing failed for {pdf_path}: {ocr_error}")
            
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
