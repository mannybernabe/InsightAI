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
                timeout=30.0
            )
            self.model = "deepseek-r1-distill-llama-70b"  
            self.search_manager = SearchManager()
            logger.info("Initialized Groq client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise

    def extract_thinking_tags(self, text: str) -> tuple[str, Optional[str]]:
        """Extract content from <think> tags and return both thinking and cleaned response."""
        thinking = None
        clean_text = text

        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        return clean_text, thinking

    @handle_rate_limit
    def generate_response(self, messages: List[Dict]) -> str:
        """Generates a response using the Groq API with retries and reasoning display."""
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

                # Add system message to encourage natural paragraph-based reasoning
                system_message = {
                    "role": "system",
                    "content": """You are a thoughtful AI assistant that explains your reasoning process naturally and clearly. For every response:

1. Write your thoughts in clear, well-spaced paragraphs under a <think> tag
2. Start with "Okay, so the user is asking..." and explain your approach
3. Break down your reasoning into natural thoughts
4. If you need to check or correct something, explain that process
5. Use a conversational tone throughout

Example format:

<think>
Okay, so the user is asking about X. Let me break this down carefully.

I'll first examine the key aspects of the question to ensure I understand what's being asked.

Now that I understand the core question, let me analyze each part step by step.

After considering all these factors, I can now formulate a clear response.
</think>

[Your final response here]"""
                }

                # Add system message to the beginning of the conversation
                messages_with_system = [system_message] + messages

                # Generate response with retry logic
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_with_system,
                    temperature=0.6,  # Adjusted for more focused responses
                    max_tokens=2000,
                    top_p=0.95,  # Added for better response quality
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
        return self.generate_response(messages)

    def determine_search_topic(self, query: str) -> Dict:
        """
        Determine appropriate search parameters based on query context.
        """
        # First, ask the LLM to analyze the query
        system_message = {
            "role": "system",
            "content": """Analyze the given query and determine the appropriate search parameters. 
            Return a JSON object with the following format:
            {
                "topic": "news" or "general",
                "days": number (only if topic is news, 1-30),
                "reasoning": "explanation of why this topic was chosen"
            }

            Use "news" for:
            - Recent events, sports games, current affairs
            - Breaking news, latest updates
            - Current market or business updates

            Use "general" for:
            - Historical information
            - Conceptual questions
            - How-to queries
            - General knowledge
            """
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    system_message,
                    {"role": "user", "content": query}
                ],
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"Search topic determination: {result}")
            return result
        except Exception as e:
            logger.error(f"Error determining search topic: {e}")
            return {"topic": "general", "reasoning": "Failed to determine topic, using default"}

    def generate_response_with_search(self, messages: List[Dict]) -> str:
        """Generates a response using web search results for enhanced accuracy."""
        try:
            if not messages or not isinstance(messages, list):
                return "Invalid message format. Please try again."

            # Get the user's query from the last message
            query = messages[-1]["content"].strip()
            if not query:
                return "Please enter a message to start the conversation."

            # Determine search parameters
            search_params = self.determine_search_topic(query)
            logger.info(f"Using search parameters: {search_params}")

            # Perform web search with better error handling
            try:
                logger.info(f"Performing Tavily search for query: {query}")
                search_results = self.search_manager.search(
                    query,
                    topic=search_params["topic"],
                    days=search_params.get("days", 3) if search_params["topic"] == "news" else None
                )
                logger.info(f"Successfully retrieved {len(search_results) if search_results else 0} results")

                if not search_results:
                    logger.warning("No search results found")
                    search_context = "No relevant search results found."
                else:
                    # Create a context from search results with proper null checks
                    search_results_text = []
                    for i, result in enumerate(search_results[:3], 1):
                        if isinstance(result, dict):
                            content = result.get('content', 'No content available')
                            url = result.get('url', '#')
                            search_results_text.append(f"Source {i}:\nContent: {content}\nURL: {url}")
                    search_context = "\n\n".join(search_results_text) if search_results_text else "No relevant search results found."
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                return self.generate_response(messages)  # Fallback to normal response

            # Add system message to encourage natural paragraph-based reasoning
            system_message = {
                "role": "system",
                "content": f"""You are a thoughtful AI assistant that explains your reasoning process naturally and clearly. 
First, analyze these search results to provide accurate, up-to-date information:

{search_context}

Search type used: {search_params["topic"].upper()} search
Reason for search type: {search_params.get("reasoning", "Not specified")}

For every response, follow these steps in order:

1. Start by examining the search results before any other reasoning
2. Use <think> tags to show your analysis
3. Begin with "Okay, so the user is asking about X. Let me analyze the search results first."
4. Reference specific information from the search results as you analyze them
5. Only after analyzing the search results, proceed with additional reasoning if needed

Example format:

<think>
Okay, so the user is asking about X. Let me analyze the search results first.

Looking at Source 1, I can see that...

Source 2 provides additional context, specifically...

After analyzing these search results, I can now formulate a comprehensive response.
</think>

[Your final response here with proper source citations]"""
            }

            # Store the raw search results for later use
            self.last_search_results = search_results

            # Add system message to the beginning of the conversation
            messages_with_system = [system_message] + messages

            try:
                # Generate response with retry logic
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_with_system,
                    temperature=0.6,
                    max_tokens=2000,
                    top_p=0.95,
                    timeout=30.0
                )

                if not response or not response.choices:
                    return "Sorry, I couldn't generate a response. Please try again."

                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                return self.generate_response(messages)  # Fallback to normal response

        except Exception as e:
            logger.error(f"Error in generate_response_with_search: {str(e)}")
            return "I encountered an error while searching. Let me try answering without search results."