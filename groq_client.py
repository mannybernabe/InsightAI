import os
import time
from groq import Groq
from typing import List, Dict
from utils import handle_rate_limit

class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        self.client = Groq(
            api_key=api_key
        )
        self.model = "deepseek-r1-distill-llama-70b"

    @handle_rate_limit
    def generate_response(self, messages: List[Dict]) -> str:
        """Generates a response using the Groq API."""
        MAX_RETRIES = 3
        RETRY_DELAY = 2

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    stream=True
                )

                # Accumulate streamed response
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content

                return full_response

            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
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

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )

                return response.choices[0].message.content

            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    error_msg = f"Error processing search after {MAX_RETRIES} attempts: {str(e)}"
                    raise Exception(error_msg)