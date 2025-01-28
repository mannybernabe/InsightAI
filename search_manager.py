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
        self.min_request_interval = 2.0  # Increased to avoid rate limiting
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

            with DDGS() as ddgs:
                logger.info(f"Performing search for query: {query}")

                # Try with web search first
                raw_results = []
                try:
                    # Use a generator to get results
                    for r in ddgs.text(
                        keywords=query,
                        region='wt-wt',
                        safesearch='off',
                        timelimit='y'  # Past year
                    ):
                        raw_results.append(r)
                        if len(raw_results) >= max_results:
                            break

                    logger.info(f"Raw results received: {len(raw_results)}")
                except Exception as e:
                    logger.error(f"Error during search: {str(e)}")
                    return []

                # Update last request time
                self.last_request_time = time.time()

                if not raw_results:
                    logger.warning("No results found")
                    return []

                formatted_results = []
                for result in raw_results:
                    if isinstance(result, dict):
                        title = result.get('title', '').strip()
                        link = result.get('link', '').strip()
                        snippet = result.get('body', '').strip()

                        logger.debug(f"Processing result: {title}")

                        if all([title, link, snippet]):
                            formatted_results.append({
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })

                            if len(formatted_results) >= max_results:
                                break

                logger.info(f"Returning {len(formatted_results)} formatted results")
                return formatted_results[:max_results]

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []