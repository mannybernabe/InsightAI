import streamlit as st
from typing import List, Dict
from groq_client import GroqClient
from utils import manage_chat_history, format_message, search_messages

class ChatInterface:
    def __init__(self):
        try:
            self.groq_client = GroqClient()
            print("GroqClient initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize GroqClient: {str(e)}")
            print(f"GroqClient initialization error: {str(e)}")
            raise

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            print("Initialized empty messages list in session state")
        if "search_results" not in st.session_state:
            st.session_state.search_results = []
        if "message_key" not in st.session_state:
            st.session_state.message_key = 0

    def handle_message(self, message: str) -> None:
        """Handle new message from user."""
        if not message.strip():
            return

        print(f"Processing new message: {message[:50]}...")
        try:
            # Add user message to history
            user_message = format_message("user", message)
            st.session_state.messages = manage_chat_history(
                st.session_state.messages,
                user_message
            )

            # Format messages for API
            messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                } for msg in st.session_state.messages
            ]

            print(f"Sending {len(messages)} messages to Groq API")

            # Get response from Groq
            response = self.groq_client.generate_response(messages)
            print(f"Received response from Groq: {response[:50]}...")

            # Add assistant response to history
            st.session_state.messages = manage_chat_history(
                st.session_state.messages,
                format_message("assistant", response)
            )

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"Error in handle_message: {error_msg}")
            st.error(error_msg)

    def handle_search(self, query: str) -> None:
        """Handle search request."""
        if not query.strip():
            st.session_state.search_results = []
            return

        try:
            # Search messages
            results = search_messages(st.session_state.messages, query)
            print(f"Found {len(results)} search results for query: {query}")

            # Generate summary if results found
            if results:
                try:
                    summary = self.groq_client.generate_search_response(query, results)
                    results.insert(0, format_message("system", f"Summary: {summary}"))
                except Exception as e:
                    print(f"Error generating search summary: {str(e)}")
                    results.insert(0, format_message("system", f"Error generating summary: {str(e)}"))

            st.session_state.search_results = results

        except Exception as e:
            error_msg = f"Error processing search: {str(e)}"
            print(f"Error in handle_search: {error_msg}")
            st.error(error_msg)

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

        # Main chat area
        st.markdown("### Chat")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(f"Sent at {message['timestamp']}")

        # Increment message key to force refresh of chat input
        st.session_state.message_key += 1

        # Process new messages with unique key
        if message := st.chat_input("Type your message here...", key=f"chat_input_{st.session_state.message_key}"):
            print("Received new chat input")
            self.handle_message(message)