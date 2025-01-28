from typing import List, Dict
import time
from datetime import datetime
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    message: Dict
    relevance_score: float
    matched_terms: List[str]

def calculate_relevance_score(query: str, content: str) -> float:
    """Calculate relevance score using sequence matcher."""
    return SequenceMatcher(None, query.lower(), content.lower()).ratio()

def parse_time_filter(time_filter: Optional[str] = None) -> Optional[tuple]:
    """Parse time filter string into start and end timestamps."""
    if not time_filter:
        return None

    try:
        if time_filter == "last_hour":
            start_time = time.time() - 3600
            return (start_time, None)
        elif time_filter == "last_day":
            start_time = time.time() - 86400
            return (start_time, None)
        elif time_filter == "last_week":
            start_time = time.time() - 604800
            return (start_time, None)
        return None
    except Exception:
        return None

def search_messages(
    history: List[Dict],
    query: str,
    role_filter: Optional[str] = None,
    time_filter: Optional[str] = None,
    min_relevance: float = 0.3
) -> List[SearchResult]:
    """Enhanced search through message history with filters and relevance scoring."""
    if not query or not history:
        return []

    # Parse filters
    time_range = parse_time_filter(time_filter)
    query_terms = query.lower().split()
    results = []

    for msg in history:
        # Apply role filter if specified
        if role_filter and msg["role"] != role_filter:
            continue

        # Apply time filter if specified
        if time_range:
            msg_time = datetime.strptime(msg["timestamp"], "%H:%M:%S").timestamp()
            start_time, end_time = time_range
            if msg_time < start_time or (end_time and msg_time > end_time):
                continue

        content = msg["content"].lower()

        # Calculate relevance score
        relevance = calculate_relevance_score(query, content)

        # Track matched terms
        matched_terms = [term for term in query_terms if term in content]

        # Add to results if meets minimum relevance or has exact matches
        if relevance >= min_relevance or matched_terms:
            results.append(SearchResult(
                message=msg,
                relevance_score=relevance,
                matched_terms=matched_terms
            ))

    # Sort by relevance score
    results.sort(key=lambda x: x.relevance_score, reverse=True)
    return results

def format_message(role: str, content: str) -> Dict:
    """Formats a message for the chat history."""
    return {
        "role": role,
        "content": content,
        "timestamp": time.strftime("%H:%M:%S")
    }

def manage_chat_history(history: List[Dict], new_message: Dict) -> List[Dict]:
    """Manages the chat history by adding new messages and maintaining size."""
    MAX_HISTORY = 50
    history.append(new_message)
    return history[-MAX_HISTORY:]

def handle_rate_limit(func):
    """Decorator to handle API rate limits."""
    def wrapper(*args, **kwargs):
        MAX_RETRIES = 3
        RETRY_DELAY = 2

        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise e
    return wrapper