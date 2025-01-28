import os
import time
import json
from openai import OpenAI
from typing import List, Dict
from search_manager import SearchManager
from utils import handle_rate_limit

class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client. Please verify your API key. Error: {str(e)}")

        self.model = "mixtral-8x7b-32768"
        self.search_manager = SearchManager()

    @handle_rate_limit
    def generate_response(self, messages: List[Dict]) -> str:
        """Generates a response using the Groq API."""
        try:
            # Check if this is a web search request
            last_message = messages[-1]["content"].lower().strip()
            if last_message.startswith("!search "):
                search_query = last_message[8:].strip()
                print(f"Processing web search request for: {search_query}")

                if not search_query:
                    return (
                        "Please provide a search query after the !search command.\n"
                        "Example: !search latest AI developments"
                    )

                # Add delay between searches
                time.sleep(3)
                search_results = self.search_manager.search(search_query)
                print(f"Search complete, found {len(search_results)} results")

                if not search_results:
                    return (
                        "No results found. Please try:\n"
                        "1. Using simpler search terms (1-2 keywords)\n"
                        "2. Waiting a few minutes before searching again\n"
                        "3. Removing special characters\n\n"
                        "Example: Instead of '!search latest artificial intelligence developments 2024', "
                        "try '!search AI news'"
                    )

                # Format search results
                results_text = "Here are the search results:\n\n"
                for i, result in enumerate(search_results, 1):
                    if result['link'] == '#':  # Error message
                        return result['snippet']

                    results_text += f"{i}. {result['title']}\n"
                    results_text += f"   {result['snippet']}\n"
                    results_text += f"   Link: {result['link']}\n\n"

                return results_text

            # Regular chat response
            formatted_messages = []
            if not any(msg["role"] == "system" for msg in messages):
                formatted_messages.append({
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. To search the web, use the '!search' "
                        "command followed by 1-2 keywords. For example: '!search AI news'. "
                        "If search fails, try again with fewer words after waiting a few minutes."
                    )
                })

            formatted_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )

            if not response.choices:
                raise Exception("No response received from API")

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
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
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Error processing search: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)