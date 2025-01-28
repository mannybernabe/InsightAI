import os
import gradio as gr
from groq_client import GroqClient
import logging
from typing import Tuple, List

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
            self.initialization_error = str(e)
            self.groq_client = None

    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """Main chat function that handles message processing with proper error handling."""
        try:
            if not message.strip():
                return history, ""

            if self.groq_client is None:
                error_msg = f"Chat system is not properly initialized. Error: {self.initialization_error}"
                logger.error(error_msg)
                return history + [(message, f"❌ {error_msg}")], ""

            logger.info(f"Processing message: {message}")

            # Convert history to messages format
            messages = []
            for h in history:
                messages.extend([
                    {"role": "user", "content": h[0]},
                    {"role": "assistant", "content": h[1]}
                ])
            messages.append({"role": "user", "content": message})

            # Get response with error handling
            response = self.groq_client.generate_response(messages)
            if response:
                logger.info("Successfully received response from Groq")
                return history + [(message, response)], ""
            else:
                error_msg = "Failed to get a valid response from the AI service"
                logger.error(error_msg)
                return history + [(message, f"❌ {error_msg}")], ""

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return history + [(message, f"❌ {error_msg}")], ""

def create_interface():
    try:
        logger.info("Creating Gradio interface...")
        chat = GradioChat()

        with gr.Blocks(title="AI Research Assistant") as demo:
            gr.Markdown("""
            # 🔍 AI Research Assistant
            Ask anything and get detailed responses with citations and reasoning.
            """)

            chatbot = gr.Chatbot(
                [],
                show_label=False,
                container=True,
                height=450,
                avatar_images=("👤", "🤖")
            )

            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Ask anything...",
                    container=False,
                    scale=8
                )
                submit = gr.Button("Send", scale=1)

            # Set up event handlers with error handling
            def on_submit(message, history):
                try:
                    return chat.chat(message, history)
                except Exception as e:
                    logger.error(f"Error in submit handler: {str(e)}")
                    return history + [(message, f"❌ Error: {str(e)}")], ""

            submit.click(
                on_submit,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            ).then(lambda: "", None, msg)

            msg.submit(
                on_submit,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            ).then(lambda: "", None, msg)

        return demo

    except Exception as e:
        error_msg = f"Error creating interface: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting Gradio interface...")
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=3000,
            share=False
        )
    except Exception as e:
        logger.error(f"Failed to start Gradio app: {str(e)}")
        raise