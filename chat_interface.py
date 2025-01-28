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
        if "processing" not in st.session_state:
            st.session_state.processing = False

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        message = format_message(role, content)
        st.session_state.messages = manage_chat_history(
            st.session_state.messages,
            message
        )

    def on_message_submit(self, user_message: str) -> None:
        """Callback for when a message is submitted."""
        if user_message.strip():
            self.add_message("user", user_message)
            st.session_state.processing = True
            st.session_state.pending_message = user_message

    def process_pending_message(self) -> None:
        """Process any pending message in the session state."""
        if hasattr(st.session_state, 'pending_message') and st.session_state.processing:
            try:
                messages = [
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    } for msg in st.session_state.messages
                ]

                # Get response from Groq
                response = self.groq_client.generate_response(messages)
                self.add_message("assistant", response)
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"Error in process_pending_message: {error_msg}")
                st.error(error_msg)
            finally:
                st.session_state.processing = False
                delattr(st.session_state, 'pending_message')

    def handle_search(self, query: str) -> None:
        """Handle search request."""
        if not query.strip():
            st.session_state.search_results = []
            return

        try:
            results = search_messages(st.session_state.messages, query)
            print(f"Found {len(results)} search results for query: {query}")

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

        # Process any pending message first
        self.process_pending_message()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(f"Sent at {message['timestamp']}")

        # Show processing indicator
        if st.session_state.processing:
            with st.chat_message("assistant"):
                st.write("Thinking...")

        # Chat input
        if message := st.chat_input("Type your message here...", disabled=st.session_state.processing):
            self.on_message_submit(message)