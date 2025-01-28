import os
import time
import json
import re
from openai import OpenAI
from typing import List, Dict, Generator, Union, Optional
from search_manager import SearchManager
from utils import handle_rate_limit
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=30.0  # Add timeout
            )
            self.model = "mixtral-8x7b-32768"
            self.search_manager = SearchManager()
            logger.info("Initialized Groq client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise

    def extract_thinking_tags(self, text: str) -> tuple[str, Optional[str]]:
        """Extract content from <think> tags and return both thinking and cleaned response."""
        thinking = None
        clean_text = text

        # Find content within <think> tags
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Remove the think tags and their content from the response
            clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        return clean_text, thinking

    @handle_rate_limit
    def generate_response(self, messages: List[Dict]) -> str:
        """Generates a response using the Groq API with retries."""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                if not messages or not isinstance(messages, list):
                    return "Invalid message format. Please try again."

                # Get the user's query from the last message
                query = messages[-1]["content"].strip()
                if not query:
                    return "Please enter a message to start the conversation."

                # Generate response with retry logic
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                    timeout=30.0
                )

                if not response or not response.choices:
                    return "Sorry, I couldn't generate a response. Please try again."

                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                return f"I'm having trouble connecting right now. Please try again in a moment. Error: {str(e)}"

        return "Service is temporarily unavailable. Please try again later."

    def generate_reasoning_stream(self, messages: List[Dict]) -> Generator:
        """Generates a streamed response showing the reasoning process."""
        return self.generate_response(messages, stream=True)