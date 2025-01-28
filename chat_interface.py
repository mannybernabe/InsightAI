import streamlit as st
from typing import List, Dict, Optional, Tuple
from groq_client import GroqClient
from utils import manage_chat_history, format_message
import re

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
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "current_response" not in st.session_state:
            st.session_state.current_response = ""

    def extract_think_tags(self, text: str) -> Tuple[str, Optional[str]]:
        """Extract content from <think> tags and return both thinking and cleaned response."""
        thinking = None
        clean_text = text

        # Find content within <think> tags
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Remove the think tags and their content from the response
            clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        return clean_text, thinking

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

                # Create containers for different parts of the response
                with st.chat_message("assistant"):
                    # Create containers
                    typing_placeholder = st.empty()
                    thinking_container = st.container()
                    response_container = st.container()

                    # Initialize the stream
                    typing_placeholder.markdown("🤔 Thinking...")
                    response_stream = self.groq_client.generate_reasoning_stream(messages)

                    # Process the stream
                    full_response = ""
                    current_thinking = ""
                    in_think_tag = False

                    for chunk in response_stream:
                        if not chunk.choices[0].delta.content:
                            continue

                        content = chunk.choices[0].delta.content
                        full_response += content

                        # Check for think tags
                        if "<think>" in content:
                            in_think_tag = True
                            with thinking_container:
                                st.markdown("### 🧠 Reasoning Process")
                                st.write("---")
                        elif "</think>" in content:
                            in_think_tag = False
                            with thinking_container:
                                st.markdown(current_thinking)
                                st.write("---")
                        elif in_think_tag:
                            current_thinking += content
                            with thinking_container:
                                st.markdown(current_thinking + "▌")
                        else:
                            # Regular response content
                            clean_response, _ = self.extract_think_tags(full_response)
                            if clean_response.strip():
                                with response_container:
                                    st.markdown(
                                        self.format_message_with_citations(clean_response) + "▌",
                                        unsafe_allow_html=True
                                    )

                    # Final update
                    typing_placeholder.empty()
                    clean_response, final_thinking = self.extract_think_tags(full_response)

                    # Update thinking container one last time if needed
                    if final_thinking:
                        with thinking_container:
                            st.markdown("### 🧠 Reasoning Process")
                            st.markdown(final_thinking)
                            st.write("---")

                    # Update response container
                    with response_container:
                        st.markdown(
                            self.format_message_with_citations(clean_response),
                            unsafe_allow_html=True
                        )

                    st.session_state.current_response = clean_response

                # Add the final response to chat history
                self.add_message("assistant", st.session_state.current_response)

            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"Error in process_pending_message: {error_msg}")
                st.error(error_msg)
            finally:
                st.session_state.processing = False
                if hasattr(st.session_state, 'pending_message'):
                    delattr(st.session_state, 'pending_message')

    def on_message_submit(self, user_message: str):
        """Handle message submission."""
        if user_message.strip():
            self.add_message("user", user_message)
            st.session_state.processing = True
            st.session_state.pending_message = user_message
            st.session_state.current_response = ""
            st.rerun()

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
                    st.write("🤔 Analyzing your query...")

            # Chat input
            if message := st.chat_input(
                "Ask anything...",
                disabled=st.session_state.processing
            ):
                self.on_message_submit(message)

        # Info section at the bottom
        st.markdown("""
            <div style='margin-top: 2rem; padding: 1rem; background-color: #f7f7f7; border-radius: 5px;'>
                <h4>Tips:</h4>
                <ul>
                    <li>Ask any question to search and analyze information</li>
                    <li>Watch the reasoning process unfold in real-time</li>
                    <li>Click on citations [1], [2], etc. to see sources</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)