import streamlit as st
from chat_interface import ChatInterface

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for Perplexity-like styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main {
            padding: 2rem;
        }
        .stMarkdown {
            font-size: 1rem;
        }
        .citation {
            color: #4a90e2;
            font-size: 0.8rem;
        }
        .source-link {
            color: #666;
            font-size: 0.9rem;
            text-decoration: none;
        }
        .source-link:hover {
            text-decoration: underline;
        }
        .search-header {
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize chat interface
    chat = ChatInterface()
    chat.initialize_session_state()

    # Header with clean, modern design
    st.markdown("""
        <div class="search-header">
            <h1>AI Research Assistant</h1>
            <p>Powered by Groq & Tavily - Ask anything or search the web with "!search your query"</p>
        </div>
    """, unsafe_allow_html=True)

    # Create and display the chat interface
    chat.create_interface()

if __name__ == "__main__":
    main()