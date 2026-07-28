import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update imports
content = content.replace('from app.agent import load_agent', 'from app.with_ollama_agent import load_agent')

# 2. Update Advanced Settings expander
old_settings = """    # Advanced Settings (Model Selection)
    with st.expander("⚙️ Advanced Settings"):
        model_name = st.selectbox(
            "AI Model", 
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it", 
                "deepseek-r1-distill-llama-70b"
            ],
            label_visibility="collapsed"
        )"""

new_settings = """    # Advanced Settings (Provider & Model Selection)
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
            ollama_base_url = st.text_input("Ollama Base URL:", value="http://localhost:11434")"""

content = content.replace(old_settings, new_settings)

# 3. Update load_agent calls (there are two)
old_load = 'st.session_state.agent = load_agent(model_name=model_name)'
new_load = '''if model_provider == "groq":
                    st.session_state.agent = load_agent(model_name=model_name, provider="groq")
                else:
                    st.session_state.agent = load_agent(model_name=model_name, provider="ollama", ollama_base_url=ollama_base_url)'''

content = content.replace(old_load, new_load)

# 4. Update system status
old_status = '''    if "agent" in st.session_state:
        st.markdown(f"<div style='text-align: center; color: var(--pale); font-size: 13px;'><span style='color: #6ebe48;'>🟢 Active</span> &nbsp;|&nbsp; 📚 {len(st.session_state.uploaded_pdfs)} Docs &nbsp;|&nbsp; 🧠 {model_name.split('-')[0].capitalize()}</div>", unsafe_allow_html=True)'''

new_status = '''    if "agent" in st.session_state:
        st.markdown(f"<div style='text-align: center; color: var(--pale); font-size: 13px;'><span style='color: #6ebe48;'>🟢 Active</span> &nbsp;|&nbsp; 📚 {len(st.session_state.uploaded_pdfs)} Docs &nbsp;|&nbsp; 📡 {provider_name} &nbsp;|&nbsp; 🧠 {model_name.split('-')[0].split(':')[0].capitalize()}</div>", unsafe_allow_html=True)'''

content = content.replace(old_status, new_status)

with open('with_ollama_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
