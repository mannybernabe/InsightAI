import streamlit as st
from typing import List, Dict
from groq_client import GroqClient
from utils import manage_chat_history, format_message, search_messages

class ChatInterface:
    def __init__(self):
        self.groq_client = GroqClient()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "search_results" not in st.session_state:
            st.session_state.search_results = []

    def handle_message(self, message: str) -> None:
        """Handle new message from user."""
        try:
            # Format messages for API
            messages = [msg for msg in st.session_state.messages]
            messages.append(format_message("user", message))

            # Get response from Groq
            response = self.groq_client.generate_response(messages)

            # Add messages to history
            st.session_state.messages = manage_chat_history(
                st.session_state.messages,
                format_message("user", message)
            )
            st.session_state.messages = manage_chat_history(
                st.session_state.messages,
                format_message("assistant", response)
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")

    def handle_search(self, query: str) -> None:
        """Handle search request."""
        if not query.strip():
            st.session_state.search_results = []
            return

        # Search messages
        results = search_messages(st.session_state.messages, query)

        # Generate summary if results found
        if results:
            try:
                summary = self.groq_client.generate_search_response(query, results)
                results.insert(0, format_message("system", f"Summary: {summary}"))
            except Exception as e:
                results.insert(0, format_message("system", f"Error generating summary: {str(e)}"))

        st.session_state.search_results = results

    def create_interface(self):
        """Creates the chat interface using Streamlit components."""
        st.sidebar.title("Search Messages")
        search_query = st.sidebar.text_input("Enter search term")
        if search_query:
            self.handle_search(search_query)

        if st.session_state.search_results:
            st.sidebar.markdown("### Search Results")
            for msg in st.session_state.search_results:
                with st.sidebar.container():
                    st.markdown(f"**{msg['role'].title()}** ({msg['timestamp']})")
                    st.markdown(msg['content'])
                    st.markdown("---")

        st.markdown("### Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(f"Sent at {message['timestamp']}")

        # Chat input
        if message := st.chat_input("Type your message here..."):
            self.handle_message(message)
