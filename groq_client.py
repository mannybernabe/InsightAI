import os
import time
from openai import OpenAI
from typing import List, Dict
from utils import handle_rate_limit

class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"  # Updated endpoint path
            )
            # Test connection with a simple request
            self.client.models.list()
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client. Please verify your API key. Error: {str(e)}")

        self.model = "mixtral-8x7b-32768"  # Updated to use a supported Groq model

    @handle_rate_limit
    def generate_response(self, messages: List[Dict]) -> str:
        """Generates a response using the Groq API."""
        MAX_RETRIES = 3
        RETRY_DELAY = 2

        for attempt in range(MAX_RETRIES):
            try:
                # Convert messages to the format expected by Groq API
                formatted_messages = []

                # Add system message if not present
                if not any(msg["role"] == "system" for msg in messages):
                    formatted_messages.append({
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    })

                # Add user messages
                formatted_messages.extend([
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                    for msg in messages
                ])

                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=1000,
                    stream=True
                )

                # Accumulate streamed response
                full_response = ""
                for chunk in chat_completion:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content

                if not full_response:
                    raise Exception("Empty response received from API")

                return full_response

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                error_msg = f"Error generating response after {MAX_RETRIES} attempts: {str(e)}"
                raise Exception(error_msg)

    @handle_rate_limit
    def generate_search_response(self, query: str, context: List[Dict]) -> str:
        """Generates a response for search results."""
        MAX_RETRIES = 3
        RETRY_DELAY = 2

        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are helping to search through chat history. "
                        "Analyze the provided context and answer the query concisely."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\nContext: {str(context)}"
                    }
                ]

                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )

                return chat_completion.choices[0].message.content

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                error_msg = f"Error processing search after {MAX_RETRIES} attempts: {str(e)}"
                raise Exception(error_msg)