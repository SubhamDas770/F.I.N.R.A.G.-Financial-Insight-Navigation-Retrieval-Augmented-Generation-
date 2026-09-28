import json
from typing import List, Dict, Optional
import redis
import tiktoken

class RedisChatMemory:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        ttl_seconds: int = 86400,  # 24 hours session expiry
        max_tokens: int = 2000,    # Max tokens to keep in context window
    ):
        """
        Manages conversational memory in Redis with automatic token-based trimming.
        """
        self.ttl = ttl_seconds
        self.max_tokens = max_tokens
        
        # Using tiktoken (cl100k_base) for fast, approximate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
        )
        try:
            self.redis.ping()
            self.enabled = True
            print("✅ Redis Chat Memory connected.")
        except redis.ConnectionError:
            print("⚠️ Redis unavailable. Chat Memory running in local dict fallback mode.")
            self.enabled = False
            self.memory_fallback: Dict[str, List[str]] = {}

    def _get_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def _count_tokens(self, text: str) -> int:
        """Returns the approximate token count of a string."""
        return len(self.encoding.encode(text))

    def add_message(self, session_id: str, role: str, content: str):
        """
        Appends a new message to the session history.
        Role should be 'user' or 'assistant'.
        """
        key = self._get_key(session_id)
        message = json.dumps({"role": role, "content": content})

        if self.enabled:
            # Push to the right end of the Redis list
            self.redis.rpush(key, message)
            # Reset TTL so active conversations don't expire
            self.redis.expire(key, self.ttl)
        else:
            if key not in self.memory_fallback:
                self.memory_fallback[key] = []
            self.memory_fallback[key].append(message)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the chat history, trimming the oldest messages if the 
        total token count exceeds max_tokens.
        """
        key = self._get_key(session_id)
        
        if self.enabled:
            raw_messages = self.redis.lrange(key, 0, -1)
        else:
            raw_messages = self.memory_fallback.get(key, [])

        if not raw_messages:
            return []

        # Parse messages from JSON
        messages = [json.loads(msg) for msg in raw_messages]
        
        # Trim history based on token count (keeping the newest messages)
        trimmed_messages = []
        current_tokens = 0

        # Traverse backwards (newest to oldest)
        for msg in reversed(messages):
            msg_tokens = self._count_tokens(msg["content"])
            if current_tokens + msg_tokens > self.max_tokens:
                break
            
            trimmed_messages.insert(0, msg)
            current_tokens += msg_tokens

        return trimmed_messages

    def clear_session(self, session_id: str):
        """Deletes a chat session from memory."""
        key = self._get_key(session_id)
        if self.enabled:
            self.redis.delete(key)
        else:
            self.memory_fallback.pop(key, None)


if __name__ == "__main__":
    # Smoke test the memory module
    memory = RedisChatMemory(max_tokens=50) # Set artificially low to test trimming
    session = "user_123"
    
    # 1. Add older messages
    memory.add_message(session, "user", "Hello, I am looking for the Q3 report.")
    memory.add_message(session, "assistant", "I can help with that. What specific metrics do you need?")
    
    # 2. Add a very long recent message (this should push the older ones out of the token window)
    long_text = "Actually, I need you to summarize the entire infrastructure cost breakdown. " * 5
    memory.add_message(session, "user", long_text)
    
    # 3. Retrieve history
    history = memory.get_history(session)
    
    print(f"\n🧠 Retrieved {len(history)} messages for session '{session}':")
    for m in history:
        print(f"[{m['role'].upper()}]: {m['content'][:60]}...")