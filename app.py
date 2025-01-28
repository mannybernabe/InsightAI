import streamlit as st
from chat_interface import ChatInterface

def main():
    # Page configuration
    st.set_page_config(
        page_title="Grok Chat Interface",
        page_icon="💬",
        layout="wide"
    )

    # Initialize chat interface
    chat = ChatInterface()
    chat.initialize_session_state()

    # Header
    st.title("Grok Chat Interface")
    st.markdown("""
    Welcome to the Grok-powered chat interface! You can:
    - Chat with the AI using the message input
    - Search through message history using the search box
    - View search results and summaries
    """)

    # Create and display the chat interface
    chat_interface = chat.create_interface()
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the Gradio interface in Streamlit
    st.write(chat_interface)

if __name__ == "__main__":
    main()
