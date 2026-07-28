import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_and_create_vectors(pdf_paths, vector_store_path="vectorstore"):
    """
    Load multiple PDF files and create a vectorstore with enhanced error handling
    """
    print(f"🔍 Function called with: {pdf_paths}")
    print(f"🔍 Type: {type(pdf_paths)}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    
    # Normalize to list of strings
    if isinstance(pdf_paths, str):
        pdf_paths = [os.path.normpath(pdf_paths)]
    elif isinstance(pdf_paths, list):
        pdf_paths = [os.path.normpath(p) for p in pdf_paths if isinstance(p, str)]
    else:
        raise ValueError(f"Invalid input type for pdf_paths: {type(pdf_paths)}")

    if not pdf_paths:
        raise ValueError("No valid PDF paths provided.")

    print(f"🔍 Normalized paths: {pdf_paths}")
    
    # Load documents from all PDFs
    all_documents = []
    processed_files = []
    
    for pdf_path in pdf_paths:
        print(f"🔍 Processing path: {pdf_path}")
        print(f"🔍 File exists: {os.path.exists(pdf_path)}")
        
        if os.path.exists(pdf_path):
            try:
                # Check file size and readability
                file_size = os.path.getsize(pdf_path)
                print(f"🔍 File size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"❌ File is empty: {pdf_path}")
                    continue
                
                # Test file readability
                try:
                    with open(pdf_path, 'rb') as f:
                        # Read first few bytes to ensure file is accessible
                        header = f.read(10)
                        if not header.startswith(b'%PDF'):
                            print(f"❌ File doesn't appear to be a valid PDF: {pdf_path}")
                            continue
                        print(f"✅ PDF header validation passed for: {pdf_path}")
                except Exception as read_error:
                    print(f"❌ Cannot read file {pdf_path}: {read_error}")
                    continue
                
                # Try to load the PDF with enhanced error handling
                print(f"🔄 Loading PDF with PyMuPDFLoader: {pdf_path}")
                loader = PyMuPDFLoader(pdf_path)
                
                try:
                    documents = loader.load()
                    total_chars = sum(len(doc.page_content.strip()) for doc in documents) if documents else 0
                    
                    if total_chars < 50:
                        print(f"⚠️ Very little text extracted ({total_chars} chars). PDF might be scanned. Triggering OCR fallback...")
                        raise Exception("Scanned PDF detected")
                        
                    print(f"🔍 PyMuPDFLoader returned {len(documents)} documents")
                except Exception as load_error:
                    print(f"❌ Initial load failed or scanned PDF detected for {pdf_path}: {load_error}")
                    
                    # Try alternative: OCR using pdf2image and pytesseract
                    try:
                        print(f"🔄 Attempting OCR processing for: {pdf_path}")
                        from pdf2image import convert_from_path
                        import pytesseract
                        from langchain_core.documents import Document
                        
                        # Convert PDF pages to images
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
                            # Create document manually
                            documents = [Document(
                                page_content=text_content,
                                metadata={
                                    'source': pdf_path,
                                    'source_file': os.path.basename(pdf_path)
                                }
                            )]
                            print(f"✅ OCR processing succeeded for: {pdf_path}")
                        else:
                            print(f"❌ No text content extracted using OCR from: {pdf_path}")
                            continue
                            
                    except ImportError as e:
                        print(f"❌ OCR dependencies not available: {e}. Make sure pytesseract and pdf2image are installed.")
                        continue
                    except Exception as manual_error:
                        print(f"❌ OCR processing failed for {pdf_path}: {manual_error}")
                        continue
                
                if not documents:
                    print(f"❌ No documents loaded from: {pdf_path}")
                    continue
                
                # Add source information to metadata
                for doc in documents:
                    if 'source_file' not in doc.metadata:
                        doc.metadata['source_file'] = os.path.basename(pdf_path)
                
                all_documents.extend(documents)
                processed_files.append(pdf_path)
                print(f"✅ Successfully processed: {pdf_path} ({len(documents)} documents)")
                
            except Exception as e:
                print(f"❌ Error processing PDF {pdf_path}: {str(e)}")
                print(f"❌ Error type: {type(e).__name__}")
                continue
        else:
            print(f"❌ File not found: {pdf_path}")
    
    print(f"🔍 Total documents loaded: {len(all_documents)} from {len(processed_files)} files")
    
    if not all_documents:
        # Provide more detailed error information
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
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save vectorstore
    print("🔄 Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_store_path)
    
    print(f"✅ Vectorstore created successfully with {len(chunks)} chunks from {len(processed_files)} PDF(s)")
    return vectorstore
