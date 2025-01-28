import streamlit as st
import os
from groq_client import GroqClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def main():
    st.title("🤖 AI Research Assistant")
    
    # Initialize chat state and client
    initialize_chat()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
                
                # Update message placeholder with response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

if __name__ == "__main__":
    main()
