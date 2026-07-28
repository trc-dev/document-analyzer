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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --dark:          #053736;
        --dark-hover:    #074f4d;
        --green:         #6ebe48;
        --green-hover:   #5aaa38;
        --bg:            #dbf3d0;
        --pale:          #edf9e5;
        --pale2:         #f5fef2;
        --white:         #ffffff;
        --border:        #c2e8b0;
        --border2:       #a8d896;
        --muted:         #3d6b5d;
        --gradient:         linear-gradient(135deg, #053736 0%, #074f4d 50%, #053736 100%);
        --gradient-green:   linear-gradient(135deg, #6ebe48 0%, #5aaa38 100%);
        --gradient-card:    linear-gradient(145deg, #ffffff 0%, #f5fef2 100%);
        --sh:     0 1px 3px rgba(5,55,54,.1), 0 1px 2px rgba(5,55,54,.06);
        --sh-md:  0 4px 12px rgba(5,55,54,.1), 0 2px 4px rgba(5,55,54,.06);
        --r:    8px;
        --rl:   12px;
    }

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background overrides */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg);
    }
    
    /* Topbar */
    [data-testid="stHeader"] {
        background: rgba(255,255,255,.95) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(194,232,176,.6);
        box-shadow: 0 2px 12px rgba(5,55,54,.06);
        height: 58px;
    }

    /* Sidebar overrides */
    [data-testid="stSidebar"] {
        background: var(--gradient) !important;
        box-shadow: 4px 0 20px rgba(5,55,54,.3);
    }
    
    /* Sidebar Text / Labels */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--pale) !important;
    }

    /* Keep file uploader text dark */
    [data-testid="stFileUploadDropzone"] * {
        color: var(--dark) !important;
    }

    /* Welcome Banner / Main Header */
    .main-header {
        background: var(--gradient);
        padding: 2rem;
        border-radius: var(--rl);
        box-shadow: var(--sh-md);
        margin-bottom: 2rem;
        border: 1px solid rgba(194,232,176,.7);
    }
    .main-header h1 {
        color: var(--green);
        font-weight: 900;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-header p {
        color: rgba(219,243,208,.6);
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: 5px;
    }

    /* Sidebar Content Box (User Card style) */
    .sidebar-content {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(110,190,72,.15);
        backdrop-filter: blur(4px);
        padding: 20px;
        border-radius: var(--rl);
        margin-bottom: 20px;
    }
    .sidebar-content h2 {
        color: var(--green);
        font-weight: 800;
        font-size: 14px;
        letter-spacing: 0.2px;
        margin-top: 0;
    }

    /* Primary Buttons (Gradient) */
    button[kind="primary"] {
        background: var(--gradient-green) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: var(--r) !important;
        padding: 0.5rem 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.1px !important;
        transition: all 0.15s cubic-bezier(.4,0,.2,1) !important;
        box-shadow: 0 2px 8px rgba(5,55,54,.2) !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(110,190,72,.3) !important;
    }
    button[kind="primary"]:active {
        transform: scale(0.98) !important;
    }

    /* Secondary Buttons (Outline) */
    button[kind="secondary"] {
        background: rgba(255,255,255,.8) !important;
        color: var(--dark) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: var(--r) !important;
        font-weight: 600 !important;
        backdrop-filter: blur(4px) !important;
        transition: all 0.15s cubic-bezier(.4,0,.2,1) !important;
    }
    button[kind="secondary"]:hover {
        background: var(--pale) !important;
        border-color: var(--green) !important;
        color: var(--green-hover) !important;
    }

    /* Chat Messages */
    .user-message {
        background: white;
        color: var(--dark);
        padding: 15px 20px;
        border-radius: var(--r2xl) var(--r2xl) 5px var(--r2xl);
        margin: 10px 0;
        margin-left: 20%;
        border: 1px solid rgba(194,232,176,.7);
        box-shadow: var(--sh);
    }
    .bot-message {
        background: var(--gradient-card);
        color: var(--dark);
        padding: 15px 20px;
        border-radius: var(--r2xl) var(--r2xl) var(--r2xl) 5px;
        margin: 10px 0;
        margin-right: 20%;
        border: 1px solid rgba(194,232,176,.7);
        border-left: 4px solid var(--green);
        box-shadow: var(--sh-md);
    }

    /* System Status Alerts (Sidebar) */
    [data-testid="stSidebar"] [data-testid="stAlert"],
    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: var(--white) !important;
    }

    /* Sidebar Input Labels (Radio, Select, Checkbox) */
    [data-testid="stSidebar"] [data-baseweb="radio"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p {
        color: var(--white) !important;
    }

    /* Expander Titles (Sidebar) */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        color: var(--white) !important;
        fill: var(--white) !important;
        font-weight: 700 !important;
    }

    /* Sidebar Logo Plaque */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        background: white;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15), inset 0 0 0 1px rgba(110,190,72,.3);
        margin-bottom: 25px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        max-width: 240px !important;
        width: 100% !important;
        margin: 0 auto;
        display: block;
    }

    /* PDF Count Badge */
    .pdf-count {
        background: var(--gradient-green);
        color: var(--dark);
        padding: 6px 15px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.2px;
        text-transform: uppercase;
        margin: 10px 0;
        text-align: center;
        box-shadow: var(--sh);
    }
    
    /* Document Display Row */
    .doc-row {
        padding: 5px 0px;
        margin-bottom: 5px;
    }
    .doc-name {
        font-weight: 700;
        color: var(--pale);
        word-break: break-all;
        line-height: 1.2;
    }
    .doc-size {
        font-size: 11px;
        color: rgba(255,255,255,.6);
    }

    /* Hide Form Submit Button */
    div[data-testid="stForm"] > div > div > button { display: none; }
    
    /* Hide specific Streamlit elements */
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

# Function to ensure files exist
def ensure_files_exist():
    """Ensure all uploaded files exist on disk before processing"""
    valid_files = []
    
    for pdf_info in st.session_state.uploaded_pdfs:
        pdf_path = pdf_info['path']
        
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            valid_files.append(pdf_info)
            print(f"✅ Validated file: {pdf_path}")
        else:
            print(f"❌ File missing or empty: {pdf_path}")
    
    return valid_files

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
                if not os.listdir("data"):
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
import base64

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return ""

logo_base64 = get_base64_image(os.path.join(os.path.dirname(__file__), 'assets', 'TRC-Arrow-logo.jpeg'))
logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="width: 100%; height: 100%; object-fit: contain;">' if logo_base64 else '<span style="font-size: 40px;">📊</span>'

st.markdown(f"""
<div style="background: white; border-radius: var(--rl); box-shadow: var(--sh-xl); overflow: hidden; margin-bottom: 3rem; position: relative; border: 1px solid rgba(194,232,176,.6);">
<!-- Decorative background elements -->
<div style="position: absolute; top: -50%; right: -10%; width: 500px; height: 500px; background: radial-gradient(circle, rgba(110,190,72,.1) 0%, transparent 70%); border-radius: 50%; pointer-events: none;"></div>
<div style="height: 6px; background: var(--gradient-green); width: 100%;"></div>
<div style="padding: 45px 50px; display: flex; align-items: center; justify-content: space-between; gap: 30px; position: relative; z-index: 1;">
<div style="flex: 1;">
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 22px;">
<span style="background: rgba(110,190,72,.12); color: var(--dark); padding: 7px 18px; border-radius: 20px; font-size: 13px; font-weight: 800; border: 1px solid rgba(110,190,72,.3); letter-spacing: 0.8px; text-transform: uppercase;">TRC Consulting</span>
<span style="background: var(--gradient); color: var(--pale); padding: 7px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: var(--sh); letter-spacing: 0.8px;">Enterprise AI</span>
</div>
<h1 style="color: var(--dark); font-weight: 900; font-size: 44px; letter-spacing: -1.2px; margin: 0 0 16px 0; line-height: 1.15;">
Intelligent <span style="background: var(--gradient-green); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Audit Assistant</span>
</h1>
<p style="color: var(--muted); font-size: 18px; margin: 0; line-height: 1.6; max-width: 700px; font-weight: 500;">
Securely upload workpapers, perform automated compliance reviews, and generate analytical insights instantly using advanced RAG intelligence.
</p>
</div>
<div style="width: 160px; height: 160px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 15px 35px rgba(5,55,54,.15), inset 0 0 0 1px rgba(110,190,72,.2); flex-shrink: 0; overflow: hidden; padding: 5px; position: relative; z-index: 2; transition: transform 0.3s ease;">
{logo_html}
</div>
</div>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Sidebar ---
with st.sidebar:
    # Logo - styled via CSS as a premium white plaque
    st.image(os.path.join(os.path.dirname(__file__), 'assets', 'trc-logo.gif'), width='stretch')
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Clean File upload section
    uploaded_files = st.file_uploader(
        "📄 Upload Workpapers (PDF)", 
        type="pdf",
        accept_multiple_files=True,
        label_visibility="visible"
    )
    
    # Process uploaded files silently
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [pdf['name'] for pdf in st.session_state.uploaded_pdfs]:
                try:
                    os.makedirs("data", exist_ok=True)
                    pdf_path = os.path.normpath(os.path.join("data", uploaded_file.name))
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                        f.flush()
                        os.fsync(f.fileno())
                    
                    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                        st.session_state.uploaded_pdfs.append({
                            'name': uploaded_file.name,
                            'path': pdf_path,
                            'size': len(uploaded_file.getvalue())
                        })
                        if st.session_state.vectorstore_created:
                            st.session_state.vectorstore_created = False
                            if "agent" in st.session_state:
                                del st.session_state.agent
                except Exception as e:
                    st.error(f"❌ Error saving {uploaded_file.name}: {str(e)}")
    
    # Minimal file management
    if st.session_state.uploaded_pdfs:
        with st.expander(f"📚 Manage Uploaded Documents ({len(st.session_state.uploaded_pdfs)})"):
            for i, pdf in enumerate(st.session_state.uploaded_pdfs):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"<span style='color: var(--pale); font-size: 13px; word-break: break-all;'>{pdf['name']}</span>", unsafe_allow_html=True)
                with col2:
                    if st.button("✕", key=f"delete_{i}", help="Remove"):
                        remove_pdf(i)
                        st.rerun()
            if st.button("Clear All Files", type="secondary", use_container_width=True):
                clear_all_pdfs()
                st.rerun()
    
    # Advanced Settings (Provider & Model Selection)
    with st.expander("⚙️ Advanced Settings"):
        provider = st.radio("Choose AI Provider:", ["🚀 Groq (Cloud)", "🏠 Ollama (Local)"])
        provider_name = "Groq" if "Groq" in provider else "Ollama"
        model_provider = "groq" if "Groq" in provider else "ollama"
        
        if model_provider == "groq":
            model_name = st.selectbox("Choose Groq Model", [
                "llama-3.3-70b-versatile", "llama-3.1-8b-instant", 
                "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", 
                "gemma2-9b-it", "deepseek-r1-distill-llama-70b"
            ])
            ollama_base_url = ""
        else:
            model_name = st.selectbox("Choose Ollama Model", [
                "llama3.2:3b", "llama3.1:8b", "gemma2:9b", "qwen2.5:7b",
                "mistral:7b", "codellama:7b", "phi3:3.8b", "neural-chat:7b"
            ])
            ollama_base_url = st.text_input("Ollama Base URL:", value="http://localhost:11434")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Primary Action Area
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        if st.session_state.uploaded_pdfs:
            try:
                valid_files = ensure_files_exist()
                if not valid_files:
                    st.error("❌ No valid PDF files found.")
                    st.session_state.uploaded_pdfs = []
                    st.rerun()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("<div style='color: white; font-weight: 600; text-align: center; margin-bottom: 10px;'>📄 Processing PDF documents...</div>", unsafe_allow_html=True)
                progress_bar.progress(25)
                
                pdf_paths = [pdf['path'] for pdf in valid_files if os.path.exists(pdf['path'])]
                load_pdf_and_create_vectors(pdf_paths)
                st.session_state.vectorstore_created = True
                progress_bar.progress(60)
                
                status_text.markdown("<div style='color: white; font-weight: 600; text-align: center; margin-bottom: 10px;'>🤖 Loading AI Agent...</div>", unsafe_allow_html=True)
                progress_bar.progress(80)
                if model_provider == "groq":
                    st.session_state.agent = load_agent(model_name=model_name, provider="groq")
                else:
                    st.session_state.agent = load_agent(model_name=model_name, provider="ollama", ollama_base_url=ollama_base_url)
                
                progress_bar.progress(100)
                status_text.empty()
                st.success(f"✨ System initialized with {len(pdf_paths)} document(s)!")
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
        else:
            st.error("⚠️ Please upload a PDF first.")
            
    # Update knowledge base button (only shown if already initialized)
    if st.session_state.vectorstore_created and st.session_state.uploaded_pdfs:
        if st.button("Update Knowledge Base", type="secondary", use_container_width=True):
            try:
                with st.spinner("🔄 Updating..."):
                    valid_files = ensure_files_exist()
                    pdf_paths = [pdf['path'] for pdf in valid_files if os.path.exists(pdf['path'])]
                    load_pdf_and_create_vectors(pdf_paths)
                    if model_provider == "groq":
                        st.session_state.agent = load_agent(model_name=model_name, provider="groq")
                    else:
                        st.session_state.agent = load_agent(model_name=model_name, provider="ollama", ollama_base_url=ollama_base_url)
                    st.success("✅ Knowledge base updated!")
            except Exception as e:
                st.error(f"❌ Update failed: {str(e)}")

    # Clean System Status
    st.markdown("<hr style='border-color: rgba(194,232,176,.2); margin: 20px 0;'>", unsafe_allow_html=True)
    if "agent" in st.session_state:
        st.markdown(f"<div style='text-align: center; color: var(--pale); font-size: 13px;'><span style='color: #6ebe48;'>🟢 Active</span> &nbsp;|&nbsp; 📚 {len(st.session_state.uploaded_pdfs)} Docs &nbsp;|&nbsp; 📡 {provider_name} &nbsp;|&nbsp; 🧠 {model_name.split('-')[0].split(':')[0].capitalize()}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center; color: rgba(255,255,255,.5); font-size: 13px;'>🔴 System Not Initialized</div>", unsafe_allow_html=True)

    # Debug mode (remove in production)
    if st.checkbox("🔍 Debug Mode"):
        st.markdown("### Debug Information")
        st.write(f"**Current working directory:** {os.getcwd()}")
        st.write(f"**Files in root:** {os.listdir('.')}")
        
        if os.path.exists('data'):
            data_files = os.listdir('data')
            st.write(f"**Files in data directory:** {data_files}")
            
            for file in data_files:
                file_path = os.path.join('data', file)
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                st.write(f"- {file}: {size} bytes")
        else:
            st.write("**Data directory does not exist**")
        
        st.write(f"**Session uploaded PDFs:** {len(st.session_state.uploaded_pdfs)}")
        for pdf in st.session_state.uploaded_pdfs:
            exists = os.path.exists(pdf['path'])
            size = os.path.getsize(pdf['path']) if exists else 0
            st.write(f"- {pdf['name']}: {'✅' if exists else '❌'} ({size} bytes)")

# --- Enhanced Main Chat Interface ---
if "agent" in st.session_state:
    # 1. Display chat history with enhanced styling
    if st.session_state.messages:
        st.markdown("### 💭 Conversation History")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            # Display messages from oldest to newest (Gemini/ChatGPT style)
            for i, (role, msg) in enumerate(st.session_state.messages):
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
        <div style="text-align: center; padding: 40px; background: var(--gradient-card); border: 1px solid var(--green); border-radius: var(--rl); color: var(--dark); box-shadow: var(--sh-md); margin-bottom: 30px;">
            <h3 style="color: var(--green); font-weight: 800;">🌟 Welcome to TRC Intelligent Assistant!</h3>
            <p style="color: var(--dark); font-size: 1.1rem;">Your {len(st.session_state.uploaded_pdfs)} PDF document(s) have been processed and I'm ready to answer questions.</p>
            <p style="color: var(--dark);">I can search across all your uploaded documents to provide comprehensive answers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(194,232,176,.5); margin: 30px 0;'>", unsafe_allow_html=True)

    # 2. Quick question suggestions
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        "Summarize the main points",
        "Identify any risks or issues", 
        "List important dates",
        "Explain the methodology"
    ]
    
    for i, (col, question) in enumerate(zip([col1, col2, col3, col4], quick_questions)):
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                with st.spinner("🔄 Searching across all documents..."):
                    try:
                        response = st.session_state.agent.invoke({"query": question})
                        answer = response["result"]
                        
                        st.session_state.messages.append(("user", question))
                        st.session_state.messages.append(("bot", answer))
                        
                        st.success("✨ Response generated from your document collection!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error occurred: {str(e)}")

    # 3. Input form at the bottom
    st.markdown("""
    <div style="background: var(--gradient-card); padding: 24px; border-radius: var(--rl); box-shadow: var(--sh); border: 1px solid rgba(194,232,176,.7); margin: 20px 0;">
        <h3 style="margin-top: 0; color: var(--dark); font-weight: 800; font-size: 18px;">💬 Ask Questions About Your Documents</h3>
        <p style="color: var(--green); font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 0;">💡 Press <strong style="color: var(--dark);">Enter</strong> to submit your question</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use form to capture Enter key press
    with st.form(key="search_form", clear_on_submit=True):
        user_query = st.text_input(
            "Type your question here...", 
            placeholder="Ask anything about your uploaded documents...",
            label_visibility="collapsed"
        )
        form_submitted = st.form_submit_button("Submit", type="primary")
    
    if form_submitted and user_query and user_query.strip():
        with st.spinner("🔄 Searching across all documents..."):
            try:
                response = st.session_state.agent.invoke({"query": user_query})
                answer = response["result"]
                
                st.session_state.messages.append(("user", user_query))
                st.session_state.messages.append(("bot", answer))
                
                st.success("✨ Response generated from your document collection!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 60px; background: var(--gradient); border-radius: var(--rl); color: var(--bg); margin: 40px 0; box-shadow: var(--sh-xl);">
        <h2 style="color: var(--green); font-weight: 900; font-size: 28px;">🚀 Getting Started with TRC Analysis</h2>
        <p style="font-size: 1.1rem; margin: 20px 0; color: rgba(219,243,208,.8);">Follow these simple steps to begin:</p>
        <div style="text-align: left; max-width: 500px; margin: 0 auto; background: rgba(255,255,255,.06); padding: 30px; border-radius: var(--rxl); border: 1px solid rgba(110,190,72,.15); backdrop-filter: blur(12px);">
            <p style="margin-bottom: 12px;"><span style="color: var(--green); font-weight: bold; margin-right: 8px;">1.</span> Upload one or multiple PDF documents via the sidebar</p>
            <p style="margin-bottom: 12px;"><span style="color: var(--green); font-weight: bold; margin-right: 8px;">2.</span> Review your uploaded documents list</p>
            <p style="margin-bottom: 12px;"><span style="color: var(--green); font-weight: bold; margin-right: 8px;">3.</span> Select an AI model from the dropdown</p>
            <p style="margin-bottom: 12px;"><span style="color: var(--green); font-weight: bold; margin-right: 8px;">4.</span> Click <strong>Process Documents & Load Agent</strong></p>
            <p style="margin-bottom: 0;"><span style="color: var(--green); font-weight: bold; margin-right: 8px;">5.</span> Ask questions by typing and pressing Enter!</p>
        </div>
        <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.03); border-radius: 10px; border-top: 1px solid rgba(110,190,72,.1);">
            <h4 style="color: var(--green); margin-bottom: 10px;">✨ Core Features:</h4>
            <p style="font-size: 14px; color: rgba(219,243,208,.7);">Cross-document search • Automated compliance checks • Smart insights</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 50px; color: #666;">
    <p>Built with ❤️ using Streamlit | Smart Multi-PDF RAG Assistant v4.0</p>
</div>
""", unsafe_allow_html=True)
