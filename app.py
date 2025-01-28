import streamlit as st
from chat_interface import ChatInterface

def main():
    # Page configuration
    st.set_page_config(
        page_title="Groq Chat Interface",
        page_icon="💬",
        layout="wide"
    )

    # Initialize chat interface
    chat = ChatInterface()
    chat.initialize_session_state()

    # Header
    st.title("Groq-Powered Chat Interface")
    st.markdown("""
    Welcome to the Groq-powered chat interface! You can:
    - Chat with the AI (using deepseek-r1-distill-llama-70b model)
    - Search through message history using the search box
    - View search results and summaries
    """)

    # Create and display the chat interface
    chat.create_interface()

if __name__ == "__main__":
    main()