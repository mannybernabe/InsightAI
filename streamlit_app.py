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

    if "groq_client" not in st.session_state:
        try:
            st.session_state.groq_client = GroqClient()
            logger.info("GroqClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GroqClient: {str(e)}")
            st.error(f"Error initializing chat system: {str(e)}")

def format_reasoning(reasoning: str):
    """Format the reasoning steps with proper styling."""
    steps = []
    current_step = ""

    for line in reasoning.split('\n'):
        line = line.strip()
        if line:
            if re.match(r'^\d+\.', line):
                if current_step:
                    steps.append(current_step)
                current_step = line
            elif current_step:
                current_step += " " + line
            else:
                current_step = line

    if current_step:
        steps.append(current_step)

    formatted_steps = "\n\n".join(steps)  # Add extra line break between steps

    return f"""
        <div style='
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.6;
            color: #4a5568;
        '>
            {formatted_steps}
        </div>
    """

def main():
    st.title("🤖 AI Research Assistant")

    # Initialize chat state and client
    initialize_chat()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                reasoning, answer = extract_reasoning(message["content"])
                if reasoning:
                    st.markdown("### 🧠 Reasoning Process")
                    st.markdown(format_reasoning(reasoning), unsafe_allow_html=True)
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

                # Generate response
                response = st.session_state.groq_client.generate_response(st.session_state.messages)

                # Extract and display reasoning and answer separately
                reasoning, answer = extract_reasoning(response)

                if reasoning:
                    # Clear the thinking placeholder
                    message_placeholder.empty()

                    # Show reasoning process
                    st.markdown("### 🧠 Reasoning Process")
                    st.markdown(format_reasoning(reasoning), unsafe_allow_html=True)

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