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

    def format_message_with_citations(self, content: str) -> str:
        """Format message content with clickable citations and proper styling."""
        # Check if the message contains references section
        if "References:" in content:
            # Split content into main text and references
            main_content, references = content.split("References:", 1)

            # Format citations in main content
            for i in range(10):  # Assuming max 10 citations
                citation = f"[{i}]"
                if citation in main_content:
                    main_content = main_content.replace(
                        citation,
                        f'<span class="citation">{citation}</span>'
                    )

            # Format references as clickable links
            formatted_refs = ""
            for ref in references.strip().split('\n'):
                if ref.strip():
                    # Extract URL if present
                    if ']' in ref and 'http' in ref:
                        ref_num = ref.split(']')[0] + ']'
                        url = ref.split('http')[1].strip()
                        url = 'http' + url
                        formatted_refs += f'<p>{ref_num} <a href="{url}" class="source-link" target="_blank">{url}</a></p>'
                    else:
                        formatted_refs += f'<p>{ref}</p>'

            return f"{main_content}<h4>References:</h4>{formatted_refs}"
        return content

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
            st.rerun()

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

                # Show typing indicator
                with st.chat_message("assistant"):
                    typing_placeholder = st.empty()
                    typing_placeholder.markdown("🤔 Thinking...")

                # Get response from Groq
                response = self.groq_client.generate_response(messages)
                self.add_message("assistant", response)

                # Remove typing indicator and rerun to show the response
                typing_placeholder.empty()
                st.rerun()

            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"Error in process_pending_message: {error_msg}")
                st.error(error_msg)
            finally:
                st.session_state.processing = False
                if hasattr(st.session_state, 'pending_message'):
                    delattr(st.session_state, 'pending_message')

    def create_interface(self):
        """Creates the chat interface using Streamlit components."""
        # Main chat area
        with st.container():
            # Process any pending message first
            self.process_pending_message()

            # Display chat messages with enhanced formatting
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        # Format assistant messages with citations
                        formatted_content = self.format_message_with_citations(message["content"])
                        st.markdown(formatted_content, unsafe_allow_html=True)
                    else:
                        # Display user messages normally
                        st.markdown(message["content"])
                    st.caption(f"Sent at {message['timestamp']}")

            # Show processing indicator
            if st.session_state.processing:
                with st.chat_message("assistant"):
                    st.write("🤔 Analyzing and searching...")

            # Chat input
            if message := st.chat_input(
                "Ask anything or type !search followed by your query...",
                disabled=st.session_state.processing
            ):
                self.on_message_submit(message)

        # Info section at the bottom
        st.markdown("""
            <div style='margin-top: 2rem; padding: 1rem; background-color: #f7f7f7; border-radius: 5px;'>
                <h4>Tips:</h4>
                <ul>
                    <li>Use <code>!search your query</code> to search the web</li>
                    <li>Click on citations [1], [2], etc. to see sources</li>
                    <li>Responses include real-time web search results and analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)