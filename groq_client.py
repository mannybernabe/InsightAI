import os
from groq import Groq
from typing import List, Dict
from utils import handle_rate_limit

class GroqClient:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = "deepseek-r1-distill-llama-70b"

    @handle_rate_limit
    def generate_response(self, messages: List[Dict]) -> str:
        """Generates a response using the Groq API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ],
                stream=True
            )
            
            # Accumulate streamed response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return full_response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            raise Exception(error_msg)

    @handle_rate_limit
    def generate_search_response(self, query: str, context: List[Dict]) -> str:
        """Generates a response for search results."""
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
                messages=messages
            )
            
            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Error processing search: {str(e)}"
            raise Exception(error_msg)
