import streamlit as st
from typing import List, Dict
from groq_client import GroqClient
from utils import manage_chat_history, format_message, search_messages, SearchResult

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
            st.rerun()  # Force immediate UI update

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
                st.rerun()  # Force immediate UI update for assistant response
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"Error in process_pending_message: {error_msg}")
                st.error(error_msg)
            finally:
                st.session_state.processing = False
                if hasattr(st.session_state, 'pending_message'):
                    delattr(st.session_state, 'pending_message')

    def handle_search(self, query: str, role_filter: str = None, time_filter: str = None) -> None:
        """Handle search request with enhanced filtering."""
        if not query.strip():
            st.session_state.search_results = []
            return

        try:
            # Perform enhanced search
            results = search_messages(
                st.session_state.messages,
                query,
                role_filter=role_filter,
                time_filter=time_filter
            )
            print(f"Found {len(results)} search results for query: {query}")

            if results:
                # Convert SearchResults to dict format for summary generation
                result_messages = [r.message for r in results]
                try:
                    summary = self.groq_client.generate_search_response(query, result_messages)
                    results.insert(0, SearchResult(
                        message=format_message("system", f"Summary: {summary}"),
                        relevance_score=1.0,
                        matched_terms=[]
                    ))
                except Exception as e:
                    print(f"Error generating search summary: {str(e)}")
                    results.insert(0, SearchResult(
                        message=format_message("system", f"Error generating summary: {str(e)}"),
                        relevance_score=1.0,
                        matched_terms=[]
                    ))

            st.session_state.search_results = results

        except Exception as e:
            error_msg = f"Error processing search: {str(e)}"
            print(f"Error in handle_search: {error_msg}")
            st.error(error_msg)

    def create_interface(self):
        """Creates the chat interface using Streamlit components."""
        # Advanced Search UI in sidebar
        st.sidebar.title("Advanced Search")

        # Search input
        search_query = st.sidebar.text_input("Search messages", key="search_input")

        # Filters
        role_filter = st.sidebar.selectbox(
            "Filter by role",
            options=[None, "user", "assistant"],
            format_func=lambda x: "All roles" if x is None else x.capitalize()
        )

        time_filter = st.sidebar.selectbox(
            "Time range",
            options=[None, "last_hour", "last_day", "last_week"],
            format_func=lambda x: "All time" if x is None else x.replace("_", " ").capitalize()
        )

        # Search button
        if st.sidebar.button("Search"):
            self.handle_search(search_query, role_filter, time_filter)

        # Display search results
        if st.session_state.search_results:
            st.sidebar.markdown("### Search Results")
            for result in st.session_state.search_results:
                with st.sidebar.container():
                    msg = result.message
                    st.markdown(f"**{msg['role'].title()}** ({msg['timestamp']})")
                    if result.matched_terms:
                        st.markdown(f"*Matched terms: {', '.join(result.matched_terms)}*")
                    if result.relevance_score < 1:  # Don't show for system messages
                        st.markdown(f"*Relevance: {result.relevance_score:.2%}*")
                    st.markdown(msg['content'])
                    st.markdown("---")

        # Main chat area with unique container
        with st.container():
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
            if message := st.chat_input("Type your message here...", 
                                    disabled=st.session_state.processing):
                self.on_message_submit(message)