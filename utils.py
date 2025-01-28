from typing import List, Dict
import time

def manage_chat_history(history: List[Dict], new_message: Dict) -> List[Dict]:
    """Manages the chat history by adding new messages and maintaining size."""
    MAX_HISTORY = 50
    history.append(new_message)
    return history[-MAX_HISTORY:]

def format_message(role: str, content: str) -> Dict:
    """Formats a message for the chat history."""
    return {
        "role": role,
        "content": content,
        "timestamp": time.strftime("%H:%M:%S")
    }

def search_messages(history: List[Dict], query: str) -> List[Dict]:
    """Searches through message history for matching content."""
    if not query:
        return []
    
    query = query.lower()
    return [
        msg for msg in history
        if query in msg["content"].lower()
    ]

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
