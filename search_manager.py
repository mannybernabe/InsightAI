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
                search_results = self._perform_search_with_retry(query, max_results)
                if search_results:
                    return search_results

                # If primary search fails, try alternative search
                return self._perform_alternative_search(query)

            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return [{
                    'title': 'Search Error',
                    'link': '#',
                    'snippet': ('Search service is temporarily unavailable. '
                              'Please try again in a few minutes with a simpler query. '
                              'Example: Instead of multiple words, try 1-2 key terms.')
                }]

        except Exception as e:
            logger.error(f"Fatal search error: {str(e)}")
            return [{
                'title': 'System Error',
                'link': '#',
                'snippet': 'An unexpected error occurred. Please try again later.'
            }]

    def _perform_search_with_retry(self, query: str, max_results: int) -> List[Dict]:
        """Perform search with retry mechanism"""
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                with DDGS() as ddgs:
                    raw_results = []
                    for result in ddgs.text(
                        keywords=query,
                        region='us-en',
                        safesearch='off',
                        max_results=max_results
                    ):
                        if isinstance(result, dict):
                            raw_results.append(result)
                            if len(raw_results) >= max_results:
                                break

                    self.last_request_time = time.time()
                    return self._format_results(raw_results)

            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)

        return []

    def _perform_alternative_search(self, query: str) -> List[Dict]:
        """Fallback search method with simplified parameters"""
        try:
            with DDGS() as ddgs:
                # Use more basic search parameters
                results = list(ddgs.text(
                    keywords=query,
                    region='wt-wt',  # Worldwide results
                    safesearch='off',
                    timelimit=None  # No time limit
                ))
                return self._format_results(results)
        except Exception as e:
            logger.error(f"Alternative search failed: {str(e)}")
            return []

    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """Format search results"""
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                title = result.get('title', '').strip()
                link = result.get('link', '').strip()
                snippet = result.get('body', '').strip()

                if all([title, link, snippet]):
                    formatted_results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })

        return formatted_results