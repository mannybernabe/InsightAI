import time
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchManager:
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds

    def _wait_for_rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Perform a web search with proper error handling and rate limiting.
        """
        logger.info(f"Starting search for query: {query}")
        results = []
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                with DDGS() as ddgs:
                    logger.info(f"Search attempt {attempt + 1}/{max_retries}")
                    raw_results = list(ddgs.text(
                        query,
                        max_results=max_results,
                        region='wt-wt',
                        safesearch='moderate'
                    ))
                    
                    logger.info(f"Found {len(raw_results)} raw results")
                    
                    for result in raw_results:
                        if not isinstance(result, dict):
                            continue
                            
                        title = result.get('title', '').strip()
                        link = result.get('link', '').strip()
                        snippet = result.get('body', result.get('snippet', '')).strip()
                        
                        if all([title, link, snippet]):
                            results.append({
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })
                    
                    if results:
                        logger.info(f"Successfully found {len(results)} valid results")
                        return results
                    
            except Exception as e:
                logger.error(f"Search error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
                continue
        
        logger.warning("Search completed with no results")
        return []
