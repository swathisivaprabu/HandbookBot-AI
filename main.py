import streamlit as st

import sys
import os

# Append the project root directory (2 levels up) to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.handbook_agent_v5 import answer_handbook_question, preprocess_and_setup_rag  # ✅ New import
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
    I'm your personal CUSD assistant! I can help you with:
    - 📅 School calendar events
    - 📚 Student handbook policies
    - 🏫 General school information
    """)
    
    st.markdown("---")
    
    # Quick Actions/Examples
    st.subheader("💡 Try asking me:")
    example_questions = [
        "When is winter break?",
        "What is the dress code policy?",
        "When does school start?",
        "What are the discipline rules?"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{question[:10]}"):
            # Set the question in session state to be processed
            st.session_state.sidebar_question = question
    

    st.markdown("---")
    
    # Chat Controls
    st.subheader("🛠️ Chat Controls")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        #LangChain’s ConversationalRetrievalChain SET UP SO IT CAN TAKE IN CHAT HISTORY
        #st.session_state.chat_history = []
        
    

    # Download chat history
    chat_history = "\n".join([
    f"**{msg['role'].title()}:** {msg['content']}" 
    for msg in st.session_state.get("messages", [])
    ]) 

    st.download_button(
        label="💾 Download Chat",
        data=chat_history if chat_history else "No messages yet.",
        file_name="cusd_bot_chat.txt",
        mime="text/plain"
    )

    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Session Stats")
    message_count = len(st.session_state.get("messages", []))
    user_messages = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", message_count)
    with col2:
        st.metric("Your Questions", user_messages)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    
    # Theme toggle (placeholder for future implementation)
    show_timestamps = st.checkbox("Show timestamps", value=False)
    auto_scroll = True
    auto_scroll = st.checkbox("Auto-scroll to bottom", value=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <small>
    💡 Tip: Click on example questions above to try them quickly!
    
    📧 Need help? Contact CUSD support.
    </small>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header section
st.image(BOT_AVATAR, width=80)           # Shows the image
st.title(f"Welcome to {BOT_NAME}")

st.markdown("""
This is your personal assistant. Ask me anything related to CUSD, and I'll try my best to help you!
""")


if st.button("Start Assistant"):
    if "qa_chain" not in st.session_state:
        with st.spinner("🧠 Initializing the handbook agent..."):
            try:
                st.session_state.qa_chain = preprocess_and_setup_rag()
                st.session_state.agent_ready = True  # flag for showing success
                st.success("Agent is ready!")
            except Exception as e:
                st.error(f"❌ Error during initialization: {e}")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


#LangChain’s ConversationalRetrievalChain SET UP SO IT CAN TAKE IN CHAT HISTORY
#if "chat_history" not in st.session_state:
#    st.session_state.chat_history = []

# Handle sidebar question selection
if "sidebar_question" in st.session_state:
    # Add the sidebar question to chat
    st.session_state.messages.append({
        "role": "user", 
        "content": st.session_state.sidebar_question
    })
    # Generate response (placeholder for now)
    #response = f"{BOT_NAME}: You asked - '{st.session_state.sidebar_question}'. This is a placeholder response."
    if st.session_state.sidebar_question:
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
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- PLACEHOLDER LOGIC ---
    # Replace this with actual API call using API_KEY

    #response = f"{BOT_NAME}: You asked - '{prompt}'. This is a placeholder response."
    if prompt:
        with st.spinner("Thinking..."):
            response = answer_handbook_question(prompt, st.session_state.qa_chain)
            '''
            #LangChain's ConversationalRetrievalChain SET UP SO IT CAN TAKE IN CHAT HISTORY
            response, updated_history = answer_handbook_question(
            user_question,
            st.session_state.qa_chain,
            st.session_state.chat_history
        )
            st.session_state.chat_history = updated_history  # Update memory
            '''

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Auto-scroll to bottom if enabled
    if auto_scroll:
        st.rerun()