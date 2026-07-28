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
                    print(f"🔍 PyMuPDFLoader returned {len(documents)} documents")
                except Exception as load_error:
                    print(f"❌ PyMuPDFLoader failed for {pdf_path}: {load_error}")
                    
                    # Try alternative: read as binary and create document manually
                    try:
                        print(f"🔄 Attempting manual PDF processing for: {pdf_path}")
                        import PyPDF2
                        
                        with open(pdf_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            text_content = ""
                            
                            for page_num, page in enumerate(pdf_reader.pages):
                                try:
                                    text_content += page.extract_text() + "\n"
                                except Exception as page_error:
                                    print(f"⚠️ Error reading page {page_num}: {page_error}")
                                    continue
                        
                        if text_content.strip():
                            # Create document manually
                            from langchain_core.documents import Document
                            documents = [Document(
                                page_content=text_content,
                                metadata={
                                    'source': pdf_path,
                                    'source_file': os.path.basename(pdf_path)
                                }
                            )]
                            print(f"✅ Manual PDF processing succeeded for: {pdf_path}")
                        else:
                            print(f"❌ No text content extracted from: {pdf_path}")
                            continue
                            
                    except ImportError:
                        print("❌ PyPDF2 not available for fallback processing")
                        continue
                    except Exception as manual_error:
                        print(f"❌ Manual PDF processing failed for {pdf_path}: {manual_error}")
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
