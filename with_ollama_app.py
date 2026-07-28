import streamlit as st
from app.with_ollama_retriever import load_pdf_and_create_vectors
from app.with_ollama_agent import load_agent
import os
import shutil

# Page config
st.set_page_config(
    page_title="Smart AI Agent", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Input container styling */
    .input-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    
    /* PDF file display */
    .pdf-count {
        background: linear-gradient(90deg, #56CCF2 0%, #2F80ED 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 10px 0;
        text-align: center;
    }
    
    /* Provider indicator */
    .provider-indicator {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 10px 0;
        text-align: center;
        font-size: 0.9em;
    }
    
    /* Button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Hide form submit button */
    div[data-testid="stForm"] > div > div > button {
        display: none;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []
if "vectorstore_created" not in st.session_state:
    st.session_state.vectorstore_created = False

# Function to clear all PDFs and cleanup
def clear_all_pdfs():
    """Function to properly clear all PDFs and cleanup resources"""
    try:
        # Remove all files from disk
        for pdf in st.session_state.uploaded_pdfs:
            if os.path.exists(pdf['path']):
                try:
                    os.remove(pdf['path'])
                    print(f"Deleted file: {pdf['path']}")
                except OSError as e:
                    print(f"Error deleting file {pdf['path']}: {e}")
        
        # Remove vectorstore directory if it exists
        if os.path.exists("vectorstore"):
            try:
                shutil.rmtree("vectorstore")
                print("Vectorstore directory removed")
            except OSError as e:
                print(f"Error removing vectorstore: {e}")
        
        # Remove data directory if empty
        if os.path.exists("data"):
            try:
                if not os.listdir("data"):  # If directory is empty
                    shutil.rmtree("data")
                    print("Empty data directory removed")
            except OSError as e:
                print(f"Error removing data directory: {e}")
        
        # Clear session state
        st.session_state.uploaded_pdfs = []
        st.session_state.vectorstore_created = False
        if "agent" in st.session_state:
            del st.session_state.agent
        
        return True
    except Exception as e:
        print(f"Error in clear_all_pdfs: {e}")
        return False

# Function to remove individual PDF
def remove_pdf(index):
    """Function to remove individual PDF and cleanup"""
    try:
        if 0 <= index < len(st.session_state.uploaded_pdfs):
            pdf = st.session_state.uploaded_pdfs[index]
            
            # Remove file from disk
            if os.path.exists(pdf['path']):
                try:
                    os.remove(pdf['path'])
                    print(f"Deleted file: {pdf['path']}")
                except OSError as e:
                    print(f"Error deleting file {pdf['path']}: {e}")
            
            # Remove from session state
            st.session_state.uploaded_pdfs.pop(index)
            
            # If no PDFs left, clean up everything
            if not st.session_state.uploaded_pdfs:
                clear_all_pdfs()
            else:
                # Mark vectorstore as outdated
                st.session_state.vectorstore_created = False
                if "agent" in st.session_state:
                    del st.session_state.agent
            
            return True
    except Exception as e:
        print(f"Error removing PDF at index {index}: {e}")
        return False

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Smart RAG Assistant</h1>
    <p style="font-size: 1.2rem; margin-top: 10px;">Ask intelligent questions based on multiple PDF documents</p>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h2 style="margin-top: 0;">⚙️ Configuration</h2>
        <p>Upload multiple documents and configure the AI model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Document Management")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose PDF documents", 
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDF files to create a comprehensive knowledge base"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [pdf['name'] for pdf in st.session_state.uploaded_pdfs]:
                # Ensure the 'data/' folder exists
                os.makedirs("data", exist_ok=True)
                
                # Save uploaded PDF to 'data/' directory
                pdf_path = os.path.join("data", uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Add to session state
                st.session_state.uploaded_pdfs.append({
                    'name': uploaded_file.name,
                    'path': pdf_path,
                    'size': len(uploaded_file.getvalue())
                })
                
                # Mark vectorstore as outdated when new files are added
                if st.session_state.vectorstore_created:
                    st.session_state.vectorstore_created = False
                    if "agent" in st.session_state:
                        del st.session_state.agent
    
    # Display uploaded PDFs
    if st.session_state.uploaded_pdfs:
        st.markdown(f"""
        <div class="pdf-count">
            📚 {len(st.session_state.uploaded_pdfs)} PDF(s) Uploaded
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Uploaded Documents:**")
        for i, pdf in enumerate(st.session_state.uploaded_pdfs):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 {pdf['name']}")
                st.caption(f"Size: {pdf['size']/1024:.1f} KB")
            with col2:
                if st.button("🗑️", key=f"delete_{i}", help="Remove this PDF"):
                    if remove_pdf(i):
                        st.success(f"✅ Removed {pdf['name']}")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to remove {pdf['name']}")
        
        # Clear all PDFs button
        if st.button("🗑️ Remove All PDFs", type="secondary"):
            if clear_all_pdfs():
                st.success("✅ All PDFs removed successfully!")
                st.rerun()
            else:
                st.error("❌ Error occurred while removing PDFs")
    
    # AI Provider Selection
    st.markdown("### 🤖 AI Provider Selection")
    
    provider = st.radio(
        "Choose AI Provider:",
        ["🚀 Groq (Cloud)", "🏠 Ollama (Local)"],
        help="Select between cloud-based Groq models or local Ollama models"
    )
    
    # Display current provider
    provider_name = "Groq" if "Groq" in provider else "Ollama"
    st.markdown(f"""
    <div class="provider-indicator">
        📡 Using: {provider_name}
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection based on provider
    st.markdown("### 🧠 AI Model Selection")
    
    if "Groq" in provider:
        model_name = st.selectbox(
            "Choose Groq Model", 
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it", 
                "deepseek-r1-distill-llama-70b"
            ],
            help="Select the Groq model for processing your queries"
        )
        model_provider = "groq"
    else:
        model_name = st.selectbox(
            "Choose Ollama Model", 
            [
                "llama3.2:3b",
                "llama3.1:8b",
                "gemma2:9b",
                "qwen2.5:7b",
                "mistral:7b",
                "codellama:7b",
                "phi3:3.8b",
                "neural-chat:7b"
            ],
            help="Select the Ollama model (make sure it's installed locally)"
        )
        model_provider = "ollama"
        
        # Ollama connection settings
        st.markdown("**Ollama Settings:**")
        ollama_base_url = st.text_input(
            "Ollama Base URL:", 
            value="http://localhost:11434",
            help="URL where Ollama is running"
        )
    
    # Load agent button
    st.markdown("### 🚀 Initialize System")
    if st.button("🔄 Process Documents & Load Agent", use_container_width=True):
        if st.session_state.uploaded_pdfs:
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process all PDFs
                status_text.text("📄 Processing PDF documents...")
                progress_bar.progress(25)
                
                # Get all PDF paths
                pdf_paths = [pdf['path'] for pdf in st.session_state.uploaded_pdfs]
                
                # Create vectorstore with all PDFs
                load_pdf_and_create_vectors(pdf_paths)
                st.session_state.vectorstore_created = True
                progress_bar.progress(60)
                
                # Loading agent
                status_text.text(f"🤖 Loading {provider_name} Agent...")
                progress_bar.progress(80)
                
                # Load agent with provider-specific parameters
                if model_provider == "groq":
                    st.session_state.agent = load_agent(
                        model_name=model_name, 
                        provider="groq"
                    )
                else:
                    st.session_state.agent = load_agent(
                        model_name=model_name, 
                        provider="ollama",
                        ollama_base_url=ollama_base_url
                    )
                
                # Complete
                progress_bar.progress(100)
                status_text.text("🎉 Ready to assist!")
                
                st.balloons()
                st.success(f"🎉 {provider_name} Agent loaded with {len(st.session_state.uploaded_pdfs)} PDF(s)!")
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                print(f"Detailed error: {e}")
        else:
            st.error("⚠️ Please upload at least one PDF first.")
    
    # Add more PDFs to existing vectorstore
    if st.session_state.vectorstore_created and st.session_state.uploaded_pdfs:
        if st.button("➕ Update Knowledge Base", use_container_width=True, help="Add new PDFs to existing knowledge base"):
            try:
                with st.spinner("🔄 Updating knowledge base..."):
                    # Get all PDF paths
                    pdf_paths = [pdf['path'] for pdf in st.session_state.uploaded_pdfs]
                    
                    # Recreate vectorstore with all PDFs
                    load_pdf_and_create_vectors(pdf_paths)
                    
                    # Reload agent with updated vectorstore
                    if model_provider == "groq":
                        st.session_state.agent = load_agent(
                            model_name=model_name, 
                            provider="groq"
                        )
                    else:
                        st.session_state.agent = load_agent(
                            model_name=model_name, 
                            provider="ollama",
                            ollama_base_url=ollama_base_url
                        )
                    
                    st.success("✅ Knowledge base updated successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error updating knowledge base: {str(e)}")
    
    # System status
    st.markdown("### 📊 System Status")
    if "agent" in st.session_state:
        st.success("🟢 Agent: Active")
        st.info(f"📚 Knowledge Base: {len(st.session_state.uploaded_pdfs)} document(s)")
        st.info(f"🤖 Provider: {provider_name}")
        st.info(f"🧠 Model: {model_name}")
    else:
        st.info("🔴 Agent: Not Loaded")
        if st.session_state.uploaded_pdfs:
            st.warning(f"📚 {len(st.session_state.uploaded_pdfs)} PDF(s) ready for processing")

# --- Enhanced Main Chat Interface ---
if "agent" in st.session_state:
    st.markdown("""
    <div class="input-container">
        <h3 style="margin-top: 0; color: #667eea;">💬 Ask Questions About Your Documents</h3>
        <p style="color: #666; margin-bottom: 15px;">The AI can search across all uploaded PDF documents to answer your questions.</p>
        <p style="color: #888; font-size: 0.9em; margin-bottom: 15px;">💡 Press <strong>Enter</strong> to submit your question</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use form to capture Enter key press - full width
    with st.form(key="search_form", clear_on_submit=True):
        user_query = st.text_input(
            "Type your question here...", 
            placeholder="Ask anything about your uploaded documents...",
            label_visibility="collapsed"
        )
        # Hidden submit button for form (triggered by Enter)
        form_submitted = st.form_submit_button("Submit", type="primary")
    
    # Quick question suggestions
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        "Summarize the main points",
        "What are the key findings?", 
        "List important dates",
        "Explain the methodology"
    ]
    
    for i, (col, question) in enumerate(zip([col1, col2, col3, col4], quick_questions)):
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # Process quick question immediately
                with st.spinner("🔄 Searching across all documents..."):
                    try:
                        # Get response from agent
                        response = st.session_state.agent.invoke({"query": question})
                        answer = response["result"]
                        
                        # Save to chat history
                        st.session_state.messages.append(("user", question))
                        st.session_state.messages.append(("bot", answer))
                        
                        # Success notification
                        st.success("✨ Response generated from your document collection!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error occurred: {str(e)}")
    
    # Process query when form is submitted (Enter pressed)
    if form_submitted and user_query and user_query.strip():
        with st.spinner("🔄 Searching across all documents..."):
            try:
                # Get response from agent
                response = st.session_state.agent.invoke({"query": user_query})
                answer = response["result"]
                
                # Save to chat history
                st.session_state.messages.append(("user", user_query))
                st.session_state.messages.append(("bot", answer))
                
                # Success notification
                st.success("✨ Response generated from your document collection!")
                
                # Rerun to update chat
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")
    
    # Display chat history with enhanced styling
    if st.session_state.messages:
        st.markdown("### 💭 Conversation History")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            # Display messages from newest to oldest
            for i, (role, msg) in enumerate(reversed(st.session_state.messages)):
                if role == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>🧑‍💼 You:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>🤖 Assistant:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h3>🌟 Welcome to Smart RAG Assistant!</h3>
            <p>Your {len(st.session_state.uploaded_pdfs)} PDF document(s) have been processed and I'm ready to answer questions.</p>
            <p>I can search across all your uploaded documents to provide comprehensive answers.</p>
            <p><strong>Type your question and press Enter!</strong></p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 60px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white; margin: 40px 0;">
        <h2>🚀 Getting Started with Multi-PDF Analysis</h2>
        <p style="font-size: 1.1rem; margin: 20px 0;">Follow these simple steps:</p>
        <div style="text-align: left; max-width: 500px; margin: 0 auto;">
            <p>📁 <strong>Step 1:</strong> Upload one or multiple PDF documents</p>
            <p>📚 <strong>Step 2:</strong> Review your uploaded documents list</p>
            <p>🤖 <strong>Step 3:</strong> Choose AI Provider (Groq or Ollama)</p>
            <p>🧠 <strong>Step 4:</strong> Select an AI model</p>
            <p>🔄 <strong>Step 5:</strong> Click 'Process Documents & Load Agent'</p>
            <p>💬 <strong>Step 6:</strong> Ask questions by pressing Enter!</p>
        </div>
        <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <h4>✨ Features:</h4>
            <p>• Upload multiple PDFs at once</p>
            <p>• Choose between Groq (cloud) and Ollama (local)</p>
            <p>• Switch between different AI models</p>
            <p>• Search across all documents simultaneously</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 50px; color: #666;">
    <p>Built with ❤️ using Streamlit | Smart Multi-PDF RAG Assistant v4.0</p>
</div>
""", unsafe_allow_html=True)
