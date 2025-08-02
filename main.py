import streamlit as st

import sys
import os

# Append the project directory (2 levels up) to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.handbook_agent_v5 import answer_handbook_question, preprocess_and_setup_rag, process_uploaded_file  # ✅ New import
import os
#from dotenv import load_dotenv

#streamlit run app/front_end_v4.py 

# App branding
BOT_NAME = "CUSD Elementary Handbook Agent"
BOT_AVATAR = "https://www.cusd80.com/cms/lib/AZ01001175/Centricity/Domain/3/CUSD_Logo_2014_WEB.png"  # 🤖 Emoji as symbol; replace with logo URL if needed

# Page config
st.set_page_config(
    page_title=BOT_NAME, 
    page_icon=BOT_AVATAR,
    layout="wide",  # Use wide layout for better sidebar experience
    initial_sidebar_state="expanded"
)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image(BOT_AVATAR, width=80)           # Shows the image
    st.header(BOT_NAME)
    st.markdown("---")
    
    # Bot Information
    st.subheader("ℹ️ About")
    st.markdown("""
    I'm your personal assistant! I can help you with:
    - School calendar events and dates
    - Student handbook policies and procedures  
    - Dress code and behavior guidelines
    - Academic policies and requirements
    
    📚 **Document Options:**
    - Use pre-loaded handbook & calendar
    - Upload your own PDF/TXT documents
    """)
    
    st.markdown("---")
    
    # Quick Actions/Examples
    st.subheader("💡 Try asking me:")
    example_questions = [
        "When is winter break?",
        "What is the dress code policy?", 
        "When does school start?",
        "What are the discipline rules?",
        "What are the school hours?",
        "What is the attendance policy?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_btn_{i}"):
            # Set the question in session state to be processed
            st.session_state.sidebar_question = question
    

    st.markdown("---")
    
    # Chat Controls
    st.subheader("Chat Controls")
    
    # Clear chat history
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        #LangChain’s ConversationalRetrievalChain SET UP SO IT CAN TAKE IN CHAT HISTORY
        #st.session_state.chat_history = []
        
    

    # Download chat history
    chat_history = "\n".join([
    f"**{msg['role'].title()}:** {msg['content']}" 
    for msg in st.session_state.get("messages", [])
    ]) 

    st.download_button(
        label="Download Chat",
        data=chat_history if chat_history else "No messages yet.",
        file_name="cusd_bot_chat.txt",
        mime="text/plain"
    )

    st.markdown("---")
    
    # Statistics
    st.subheader("Session Stats")
    message_count = len(st.session_state.get("messages", []))
    user_messages = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", message_count)
    with col2:
        st.metric("Your Questions", user_messages)
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    
    # Theme toggle (placeholder for future implementation)
    show_timestamps = st.checkbox("Show timestamps", value=False)
    auto_scroll = True
    auto_scroll = st.checkbox("Auto-scroll to bottom", value=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <small>
    💡 **Tip:** Click on example questions above to try them quickly!
    
    📋 **Options:** Use pre-loaded documents or upload your own PDF/TXT files.
    Initialize the assistant first before asking questions.
    </small>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header section
st.image(BOT_AVATAR, width=80)           # Shows the image
st.title(f"Welcome to {BOT_NAME}")

st.markdown("""
This is your personal assistant for CUSD Elementary Handbook and Calendar information. Ask me anything related to school policies, events, and procedures!
""")

# Create two columns for different initialization options
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Use Pre-loaded Documents")
    st.markdown("Load the default handbook and calendar documents.")
    
    if st.button("Start Assistant", key="start_preloaded"):
        if "qa_chain" not in st.session_state:
            with st.spinner("Initializing the handbook agent..."):
                try:
                    st.session_state.qa_chain = preprocess_and_setup_rag()
                    st.session_state.agent_ready = True
                    st.session_state.document_source = "pre-loaded"
                    st.success("✅ Agent is ready! I've loaded the handbook and calendar documents.")
                except Exception as e:
                    st.error(f"❌ Error during initialization: {e}")
                    st.error("Make sure the data folder contains handbook and calendar files.")

with col2:
    st.subheader("📄 Upload Your Own Documents")
    st.markdown("Upload PDF or text files to create a custom knowledge base.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file", 
        type=["pdf", "txt"],
        help="Upload a document to create a custom knowledge base"
    )
    
    if uploaded_file is not None:
        st.write(f"📁 **Selected file:** {uploaded_file.name}")
        st.write(f"📏 **File size:** {uploaded_file.size} bytes")
        
        if st.button("Process Uploaded Document", key="process_upload"):
            with st.spinner("Processing your document..."):
                try:
                    st.session_state.qa_chain = process_uploaded_file(uploaded_file)
                    if st.session_state.qa_chain:
                        st.session_state.agent_ready = True
                        st.session_state.document_source = f"uploaded: {uploaded_file.name}"
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    else:
                        st.error("❌ Failed to process the document. Please check the file format and content.")
                except Exception as e:
                    st.error(f"❌ Error processing document: {e}")

# Show current document source if agent is ready
if st.session_state.get("agent_ready", False):
    source = st.session_state.get("document_source", "unknown")
    st.info(f"🤖 **Assistant Status:** Ready | **Document Source:** {source}")

st.markdown("---")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


#LangChain’s ConversationalRetrievalChain SET UP SO IT CAN TAKE IN CHAT HISTORY
#if "chat_history" not in st.session_state:
#    st.session_state.chat_history = []

    # Handle sidebar question selection
if "sidebar_question" in st.session_state:
    # Check if assistant is ready
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        st.warning("⚠️ Please click 'Start Assistant' first to initialize the system.")
    else:
        # Add the sidebar question to chat
        st.session_state.messages.append({
            "role": "user", 
            "content": st.session_state.sidebar_question
        })
        
        with st.spinner("Thinking..."):
            response = answer_handbook_question(st.session_state.sidebar_question, st.session_state.qa_chain)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })
    # Clear the sidebar question
    del st.session_state.sidebar_question
    st.rerun()

# Display previous messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if show_timestamps and i > 0:  # Don't show timestamp for first message
            st.caption(f"Message {i+1}")
        st.markdown(message["content"])


# User input


if prompt := st.chat_input("Ask your question here..."):
    # Check if assistant is ready
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        st.warning("⚠️ Please click 'Start Assistant' first to initialize the system.")
        st.stop()
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from the handbook agent
    with st.spinner("Thinking..."):
        response = answer_handbook_question(prompt, st.session_state.qa_chain)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Auto-scroll to bottom if enabled
    if auto_scroll:
        st.rerun()