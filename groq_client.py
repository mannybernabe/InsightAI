import os
import time
import json
from openai import OpenAI
from typing import List, Dict
from duckduckgo_search import DDGS
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

    def web_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """Perform web search using DuckDuckGo with fresh session."""
        try:
            print(f"Performing web search for query: {query}")
            results = []
            retries = 3
            retry_delay = 2  # seconds between retries

            while retries > 0:
                try:
                    # Create a fresh DDGS session for each attempt
                    with DDGS() as ddgs:
                        print(f"Search attempt {4-retries} for query: {query}")
                        search_results = list(ddgs.text(
                            query,
                            max_results=num_results,
                            region='wt-wt',  # Worldwide results
                            safesearch='moderate'  # Default safe search
                        ))
                        print(f"Raw search results: {search_results}")

                        if search_results:
                            for r in search_results:
                                if isinstance(r, dict):
                                    title = r.get('title', '').strip()
                                    link = r.get('link', '').strip()
                                    body = r.get('body', r.get('snippet', '')).strip()

                                    if title and link and body:
                                        results.append({
                                            'title': title,
                                            'link': link,
                                            'snippet': body
                                        })

                            if results:
                                print(f"Successfully found {len(results)} results")
                                break

                    retries -= 1
                    if retries > 0:
                        print(f"No valid results found, retrying in {retry_delay}s... ({retries} attempts left)")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff

                except Exception as search_error:
                    print(f"Search attempt error: {str(search_error)}")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying search after error in {retry_delay}s... ({retries} attempts left)")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    continue

            if not results:
                print("No results found from DuckDuckGo search after all retries")
            return results

        except Exception as e:
            print(f"Web search error: {str(e)}")
            return []

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

                search_results = self.web_search(search_query)
                print(f"Search complete, found {len(search_results)} results")

                if not search_results:
                    return (
                        "No search results found. Please try:\n"
                        "1. A different search query\n"
                        "2. More specific terms\n"
                        "3. Using fewer words\n"
                        "4. Removing special characters\n\n"
                        "Example: Instead of '!search what's new in AI?', try '!search latest AI developments'"
                    )

                # Format search results
                results_text = "Here are the search results:\n\n"
                for i, result in enumerate(search_results, 1):
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
                        "command followed by your query (e.g. '!search latest AI developments' "
                        "or '!search python programming tutorials')."
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