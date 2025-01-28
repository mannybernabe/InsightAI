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

    def chat(self, message, history):
        """Main chat function that handles message processing."""
        if not message.strip():
            return history, ""

        try:
            logger.info(f"Processing message: {message}")

            # Convert history to messages format for GroqClient
            messages = []
            for h in history:
                messages.extend([
                    {"role": "user", "content": h[0]},
                    {"role": "assistant", "content": h[1]}
                ])

            # Add current message
            messages.append({"role": "user", "content": message})

            # Get response from GroqClient
            response = self.groq_client.generate_response(messages)
            if isinstance(response, str):
                logger.info("Got response from Groq")
                # Update history with the new message pair
                history.append((message, response))
                return history, ""
            else:
                logger.error("Unexpected response type from Groq")
                return history + [(message, "❌ Error: Unexpected response type")], ""

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return history + [(message, f"❌ Error: {str(e)}")], ""

def create_interface():
    try:
        logger.info("Creating Gradio interface...")
        chat = GradioChat()
        logger.info("Created GradioChat instance")

        with gr.Blocks() as demo:
            gr.Markdown("# 🔍 AI Research Assistant")

            chatbot = gr.Chatbot(
                [],
                show_label=False,
                container=True,
                height=400,
                avatar_images=("👤", "🤖")
            )

            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Ask anything...",
                    container=False
                )
                submit = gr.Button("Send")

            # Set up event handlers
            submit.click(
                chat.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            ).then(lambda: "", None, msg)

            msg.submit(
                chat.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            ).then(lambda: "", None, msg)

        return demo

    except Exception as e:
        logger.error(f"Error creating interface: {str(e)}")
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