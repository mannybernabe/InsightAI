import time
import logging
from typing import List, Dict, Optional
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchManager:
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 5.0  # Increased interval to reduce rate limiting
        logger.info("SearchManager initialized")

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Perform a web search with proper error handling and rate limiting.
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

            try:
                with DDGS() as ddgs:
                    logger.info(f"Performing search for query: {query}")

                    # Initialize results list
                    raw_results = []

                    # Use iterator to handle results one at a time
                    for result in ddgs.text(
                        keywords=query,
                        region='us-en',  # Use US region
                        safesearch='off',
                        timelimit='y',  # Past year
                        max_results=max_results
                    ):
                        if isinstance(result, dict):
                            raw_results.append(result)
                            if len(raw_results) >= max_results:
                                break

                    self.last_request_time = time.time()

                    if not raw_results:
                        return [{
                            'title': 'Search Unavailable',
                            'link': '#',
                            'snippet': ('The search service is currently unavailable. '
                                      'This might be due to rate limiting or temporary service issues. '
                                      'Please try again in a few minutes with a different search query.')
                        }]

                    # Format results
                    formatted_results = []
                    for result in raw_results:
                        title = result.get('title', '').strip()
                        link = result.get('link', '').strip()
                        snippet = result.get('body', '').strip()

                        if all([title, link, snippet]):
                            formatted_results.append({
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })

                    logger.info(f"Successfully retrieved {len(formatted_results)} results")
                    return formatted_results

            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return [{
                    'title': 'Search Error',
                    'link': '#',
                    'snippet': ('Failed to perform search. This might be due to rate limiting. '
                              'Please try again in a few minutes with a different search query.')
                }]

        except Exception as e:
            logger.error(f"Fatal search error: {str(e)}")
            return [{
                'title': 'System Error',
                'link': '#',
                'snippet': 'An unexpected error occurred. Please try again later.'
            }]