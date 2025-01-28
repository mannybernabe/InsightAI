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
                        "Example: !search latest AI developments in 2024"
                    )

                search_results = self.search_manager.search(search_query)
                print(f"Search complete, found {len(search_results)} results")

                if not search_results:
                    return (
                        "No results found. Please try:\n"
                        "1. Using different search terms\n"
                        "2. Making your search more specific\n"
                        "3. Checking for typos\n\n"
                        "Example: Instead of '!search ai', try '!search artificial intelligence developments 2024'"
                    )

                # Get search context first for a high-level summary
                try:
                    context = self.search_manager.get_search_context(search_query)
                except Exception as e:
                    print(f"Error getting search context: {str(e)}")
                    context = None

                # Prepare search results for analysis
                sources = []
                content_for_analysis = f"Query: {search_query}\n\n"

                if context:
                    content_for_analysis += f"Context Overview:\n{context}\n\n"

                content_for_analysis += "Detailed Sources:\n\n"

                for i, result in enumerate(search_results, 1):
                    sources.append(f"[{i}] {result['link']}")
                    content_for_analysis += f"Source [{i}]:\n"
                    content_for_analysis += f"Title: {result['title']}\n"
                    content_for_analysis += f"Content: {result['snippet']}\n\n"

                # Generate comprehensive analysis
                analysis_prompt = {
                    "role": "system",
                    "content": """You are a Perplexity-like AI research assistant. Analyze the provided search results and create a comprehensive response that:
1. Starts with a clear, direct answer to the query
2. Provides detailed analysis and insights
3. Includes specific citations using [1], [2], etc. format
4. Organizes information logically
5. Ends with a "References" section listing all sources

Keep your tone professional but conversational. Ensure all claims are backed by the sources provided."""
                }

                analysis_messages = [
                    analysis_prompt,
                    {"role": "user", "content": content_for_analysis}
                ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=analysis_messages,
                    temperature=0.7,
                    max_tokens=2000,
                    stream=False
                )

                analysis = response.choices[0].message.content

                # Add references section if not already included
                if "References:" not in analysis:
                    analysis += "\n\nReferences:\n"
                    for source in sources:
                        analysis += f"{source}\n"

                return analysis

            # Regular chat response
            formatted_messages = []
            if not any(msg["role"] == "system" for msg in messages):
                formatted_messages.append({
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant with capabilities similar to Perplexity. "
                        "To search the web, use the '!search' command followed by your query. "
                        "Example: '!search What are the latest developments in AI for 2024?'"
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