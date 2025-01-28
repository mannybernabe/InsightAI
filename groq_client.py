import os
import time
import json
import re
from openai import OpenAI
from typing import List, Dict, Generator, Union, Optional
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
    def generate_response(self, messages: List[Dict], stream: bool = False) -> Union[str, Generator]:
        """Generates a response using the Groq API with optional streaming."""
        try:
            # Get the user's query from the last message
            query = messages[-1]["content"].strip()

            # Perform web search for every query
            search_results = self.search_manager.search(query)
            print(f"Search complete, found {len(search_results)} results")

            if not search_results:
                return (
                    "I couldn't find any relevant results. Please try:\n"
                    "1. Using different search terms\n"
                    "2. Making your search more specific\n"
                    "3. Checking for typos"
                )

            # Get search context for a high-level summary
            try:
                context = self.search_manager.get_search_context(query)
            except Exception as e:
                print(f"Error getting search context: {str(e)}")
                context = None

            # Prepare search results for analysis
            sources = []
            content_for_analysis = f"Query: {query}\n\n"

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
                "content": """You are a Perplexity-like AI research assistant. For every query:
1. First, explain your thinking process inside <think> tags
2. Then provide a clear analysis that:
   - Starts with a direct answer
   - Includes detailed analysis with citations [1], [2], etc.
   - Organizes information logically
   - Ends with a "References" section

Example format:
<think>
Let me analyze the sources and synthesize a comprehensive response...
[Your reasoning process here]
</think>
[Your final response with citations]

Keep your tone professional but conversational. Back all claims with sources."""
            }

            analysis_messages = [
                analysis_prompt,
                {"role": "user", "content": content_for_analysis}
            ]

            # Generate response with optional streaming
            response = self.client.chat.completions.create(
                model=self.model,
                messages=analysis_messages,
                temperature=0.7,
                max_tokens=2000,
                stream=stream
            )

            if stream:
                return response

            analysis = response.choices[0].message.content

            # Extract thinking and clean response
            final_response, thinking = self.extract_thinking_tags(analysis)

            # Add references section if not already included
            if "References:" not in final_response:
                final_response += "\n\nReferences:\n"
                for source in sources:
                    final_response += f"{source}\n"

            return final_response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    def generate_reasoning_stream(self, messages: List[Dict]) -> Generator:
        """Generates a streamed response showing the reasoning process."""
        return self.generate_response(messages, stream=True)