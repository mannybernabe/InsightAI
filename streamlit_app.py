import streamlit as st
import os
from groq_client import GroqClient
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_reasoning(text: str):
    """Extract the reasoning section from the response."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return None, text

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "search_enabled" not in st.session_state:
        st.session_state.search_enabled = False

    if "groq_client" not in st.session_state:
        try:
            st.session_state.groq_client = GroqClient()
            logger.info("GroqClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GroqClient: {str(e)}")
            st.error(f"Error initializing chat system: {str(e)}")

def format_thinking(text):
    """Format thinking content with proper styling."""
    paragraphs = []
    current_paragraph = ""

    for line in text.strip().split('\n'):
        line = line.strip()
        if line:
            current_paragraph = current_paragraph + " " + line if current_paragraph else line
        elif current_paragraph:  # Empty line indicates paragraph break
            paragraphs.append(current_paragraph)
            current_paragraph = ""

    if current_paragraph:  # Add the last paragraph
        paragraphs.append(current_paragraph)

    formatted_paragraphs = "\n\n".join(paragraphs)  # Double line break between paragraphs

    return """<div style='
                background-color: #f0f2f6; 
                padding: 1.5rem; 
                border-radius: 8px; 
                margin-bottom: 1rem; 
                font-family: monospace; 
                white-space: pre-wrap;
                line-height: 1.6;
            '>{}</div>""".format(formatted_paragraphs)

def main():
    st.title("🤖 AI Research Assistant")

    # Initialize chat state and client
    initialize_chat()

    # Add search toggle at the top
    st.session_state.search_enabled = st.toggle(
        "Enable web search",
        help="When enabled, the AI will search the web to provide more accurate and up-to-date information.",
        value=st.session_state.search_enabled
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                reasoning, answer = extract_reasoning(message["content"])
                if reasoning:
                    st.markdown("### 🧠 Reasoning Process")
                    st.markdown(format_thinking(reasoning), unsafe_allow_html=True)
                st.markdown(answer)
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤔 Thinking...")

                # Generate response with search if enabled
                if st.session_state.search_enabled:
                    # Use search-enabled response generation
                    response = st.session_state.groq_client.generate_response_with_search(
                        st.session_state.messages
                    )
                else:
                    # Use standard response generation
                    response = st.session_state.groq_client.generate_response(
                        st.session_state.messages
                    )

                # Extract and display reasoning and answer separately
                reasoning, answer = extract_reasoning(response)

                if reasoning:
                    # Clear the thinking placeholder
                    message_placeholder.empty()

                    # Show reasoning process
                    st.markdown("### 🧠 Reasoning Process")
                    st.markdown(format_thinking(reasoning), unsafe_allow_html=True)

                # Show final response
                if answer.strip():
                    st.markdown("### Response")
                    st.markdown(answer.strip())

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

if __name__ == "__main__":
    main()