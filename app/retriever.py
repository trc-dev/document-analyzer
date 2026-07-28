import os
import concurrent.futures
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pytesseract
import fitz
from PIL import Image
import io

_GLOBAL_EMBEDDINGS = None

def get_embeddings():
    """Cache the embedding model globally so PyTorch doesn't reload it."""
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        print("⚡ Loading Embedding Model into Memory (Only happens once)...")
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'batch_size': 64},
            model_kwargs={'device': 'cpu'}
        )
    return _GLOBAL_EMBEDDINGS

def process_single_page(page_bytes, i):
    """Process a single OCR page in a separate thread/process"""
    try:
        img = Image.open(io.BytesIO(page_bytes))
        page_text = pytesseract.image_to_string(img)
        return page_text + "\n"
    except Exception as page_error:
        print(f"⚠️ Error running OCR on page {i}: {page_error}")
        return ""

def process_single_pdf(pdf_path):
    """Process a single PDF, falling back to multi-threaded OCR if needed"""
    print(f"🔍 Processing path: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return []
        
    try:
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            print(f"❌ File is empty: {pdf_path}")
            return []
            
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
            if not header.startswith(b'%PDF'):
                print(f"❌ File doesn't appear to be a valid PDF: {pdf_path}")
                return []
    except Exception as read_error:
        print(f"❌ Cannot read file {pdf_path}: {read_error}")
        return []
        
    print(f"🔄 Loading PDF with PyMuPDFLoader: {pdf_path}")
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
            print(f"🔄 Attempting multithreaded OCR processing for: {pdf_path}")
            
            # Configure Tesseract path for local Windows execution
            tesseract_path = r'C:\Users\Sonu Kumar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                
            doc = fitz.open(pdf_path)
            
            # Extract images in the main thread (PyMuPDF objects aren't thread-safe)
            page_images = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                page_images.append(pix.tobytes("png"))
            
            # Run OCR on all images in parallel
            text_content = ""
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(page_images), 8) if page_images else 1) as executor:
                results = executor.map(process_single_page, page_images, range(len(page_images)))
                for result in results:
                    text_content += result
                    
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
        except ImportError as e:
            documents = []
            print(f"❌ OCR dependencies not available: {e}. Make sure pytesseract and Pillow are installed.")
        except Exception as ocr_error:
            documents = []
            print(f"❌ OCR processing failed for {pdf_path}: {ocr_error}")
    
    # Add source information to metadata
    if documents:
        for doc in documents:
            if 'source_file' not in doc.metadata:
                doc.metadata['source_file'] = os.path.basename(pdf_path)
        print(f"✅ Successfully processed: {pdf_path} ({len(documents)} documents)")
    
    return documents

def load_pdf_and_create_vectors(pdf_paths, vector_store_path="vectorstore"):
    """
    Load multiple PDF files and create a vectorstore with enhanced error handling
    """
    print(f"🔍 Function called with: {pdf_paths}")
    
    # Normalize to list of strings
    if isinstance(pdf_paths, str):
        pdf_paths = [os.path.normpath(pdf_paths)]
    elif isinstance(pdf_paths, list):
        pdf_paths = [os.path.normpath(p) for p in pdf_paths if isinstance(p, str)]
    else:
        raise ValueError(f"Invalid input type for pdf_paths: {type(pdf_paths)}")

    if not pdf_paths:
        raise ValueError("No valid PDF paths provided.")

    all_documents = []
    
    # Process multiple PDFs in parallel
    max_workers = min(len(pdf_paths), 8) if pdf_paths else 1
    if len(pdf_paths) > 1:
        print(f"⚡ Processing {len(pdf_paths)} documents in parallel with {max_workers} threads...")
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_pdf, pdf_paths)
        for docs in results:
            all_documents.extend(docs)
    
    if not all_documents:
        error_details = []
        for path in pdf_paths:
            if os.path.exists(path):
                size = os.path.getsize(path)
                error_details.append(f"- {path}: exists ({size} bytes) but failed to process")
            else:
                error_details.append(f"- {path}: file not found")
        
        error_msg = f"No valid PDF documents found to process.\nDetails:\n" + "\n".join(error_details)
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"🔍 Created {len(chunks)} chunks")
    
    # Create embeddings - OPTIMIZED for batching and cached
    print("⚡ Retrieving cached embeddings...")
    embeddings = get_embeddings()
    
    # Create and save vectorstore
    print("🔄 Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_store_path)
    
    print(f"✅ Vectorstore created successfully with {len(chunks)} chunks from {len(pdf_paths)} PDF(s)")
    return vectorstore
