import os
import time
import logging
from typing import List, Dict, Optional
from tavily import TavilyClient
from utils import format_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchManager:
    def __init__(self):
        try:
            self.client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            self.last_request_time = 0
            self.min_request_interval = 1.0  # Tavily has better rate limits
            logger.info("SearchManager initialized with Tavily API")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {str(e)}")
            raise

    def search(self, query: str, max_results: int = 3, topic: str = "general", days: Optional[int] = None) -> List[Dict]:
        """
        Perform a web search using Tavily API with proper error handling.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
            topic: Search topic type ("general" or "news")
            days: Number of days back to search (only for news topic)
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            # Implement rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last_request
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)

            # Prepare search parameters
            search_params = {
                'query': query,
                'max_results': max_results,
                'topic': topic
            }

            # Only add days parameter for news searches
            if topic == "news" and days is not None:
                search_params['days'] = days

            logger.info(f"Performing Tavily search with params: {search_params}")
            response = self.client.search(**search_params)
            self.last_request_time = time.time()

            if not response or not response.get('results'):
                return [{
                    'title': 'No Results',
                    'link': '#',
                    'snippet': 'No search results found. Please try with different search terms.'
                }]

            # Format results to match our expected structure
            formatted_results = []
            for result in response['results'][:max_results]:
                formatted_results.append({
                    'title': result.get('title', '').strip(),
                    'url': result.get('url', '#').strip(),
                    'content': result.get('content', '').strip()
                })

            logger.info(f"Successfully retrieved {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            logger.error(error_msg)
            return [{
                'title': 'Search Error',
                'link': '#',
                'snippet': 'An error occurred while performing the search. Please try again later.'
            }]

    def get_search_context(self, query: str) -> str:
        """
        Get a summarized context for RAG applications.
        """
        try:
            context = self.client.get_search_context(query=query)
            return context
        except Exception as e:
            logger.error(f"Error getting search context: {str(e)}")
            return "Unable to generate search context at this time."