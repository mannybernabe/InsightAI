import os
import gradio as gr
from groq_client import GroqClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioChat:
    def __init__(self):
        try:
            self.groq_client = GroqClient()
            logger.info("Initialized Groq client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise

    def search(self, query: str, max_results: int = 3) -> str:
        """
        Perform a web search using Tavily API with proper error handling.
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return "Please enter a message to start the conversation."

        try:
            logger.info(f"Processing query: {query}")
            response = self.groq_client.generate_response([{"role": "user", "content": query}])
            logger.info("Successfully generated response")
            return response
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

    def chat(self, message, history):
        """Main chat function that handles message processing and streaming."""
        try:
            # Simple non-streaming response first to verify basic functionality
            logger.info(f"Processing message: {message}")
            response = self.search(message)
            logger.info("Generated response successfully")

            # Update history and return
            history.append((message, response))
            return history, ""
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            # Return the error in the chat history
            return history + [(message, f"❌ Error: {str(e)}")], ""

def create_interface():
    try:
        logger.info("Creating Gradio interface...")
        chat = GradioChat()
        logger.info("Created GradioChat instance")

        with gr.Blocks() as demo:
            gr.Markdown("# 🔍 AI Research Assistant")

            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=("👤", "🤖"),
                height=500
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask anything...",
                    show_label=False,
                    scale=9
                )
                submit = gr.Button("Send", scale=1)

            msg.submit(
                chat.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg],
                api_name=None
            ).then(lambda: "", None, msg)

            submit.click(
                chat.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg],
                api_name=None
            ).then(lambda: "", None, msg)

        return demo

    except Exception as e:
        logger.error(f"Error creating interface: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting Gradio interface...")
        demo = create_interface()
        # Removed queue() call to simplify the setup
        logger.info("Launching Gradio app...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=3000,
            share=False
        )
    except Exception as e:
        logger.error(f"Failed to start Gradio app: {str(e)}")
        raise