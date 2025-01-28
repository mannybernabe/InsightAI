import gradio as gr
from typing import List, Dict
import streamlit as st
from grok_client import GrokClient
from utils import manage_chat_history, format_message, search_messages

class ChatInterface:
    def __init__(self):
        self.grok_client = GrokClient()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "search_results" not in st.session_state:
            st.session_state.search_results = []

    def create_interface(self):
        """Creates the chat interface using Gradio components."""
        with gr.Blocks() as chat_interface:
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        height=600
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Type your message here...",
                            scale=4
                        )
                        submit = gr.Button("Send", scale=1)

                with gr.Column(scale=1):
                    search_input = gr.Textbox(
                        show_label=False,
                        placeholder="Search messages...",
                        scale=1
                    )
                    search_results = gr.JSON(
                        [],
                        label="Search Results"
                    )

            def on_message(message: str, history: List[List[str]]) -> tuple:
                if not message.strip():
                    return "", history
                
                try:
                    # Format messages for API
                    messages = [
                        format_message("user" if i % 2 == 0 else "assistant", m)
                        for conv in history
                        for i, m in enumerate(conv)
                    ]
                    messages.append(format_message("user", message))
                    
                    # Get response from Grok
                    response = self.grok_client.generate_response(messages)
                    
                    # Update history
                    history.append([message, response])
                    return "", history
                
                except Exception as e:
                    return "", history + [[message, f"Error: {str(e)}"]]

            def on_search(query: str, history: List[List[str]]) -> List[Dict]:
                if not query.strip():
                    return []
                
                # Convert history to flat message list
                messages = [
                    format_message("user" if i % 2 == 0 else "assistant", m)
                    for conv in history
                    for i, m in enumerate(conv)
                ]
                
                # Search messages
                results = search_messages(messages, query)
                
                # Generate summary if results found
                if results:
                    try:
                        summary = self.grok_client.generate_search_response(query, results)
                        results.insert(0, format_message("system", f"Summary: {summary}"))
                    except Exception as e:
                        results.insert(0, format_message("system", f"Error generating summary: {str(e)}"))
                
                return results

            # Set up event handlers
            submit.click(
                on_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            msg.submit(
                on_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            search_input.change(
                on_search,
                inputs=[search_input, chatbot],
                outputs=[search_results]
            )

        return chat_interface
