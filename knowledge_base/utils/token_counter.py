"""
Token counting utilities for OpenAI API.
"""
import tiktoken
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TokenCounter:
    """Count tokens for OpenAI API calls."""
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize token counter."""
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: list) -> int:
        """Count tokens in a list of messages."""
        num_tokens = 0
        for message in messages:
            # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            
            if "role" in message:
                num_tokens += self.count_tokens(message["role"])
                
            if "content" in message:
                num_tokens += self.count_tokens(message["content"])
                
            if "name" in message:
                num_tokens += self.count_tokens(message["name"])
                num_tokens -= 1  # role is omitted when name is present
        
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # GPT-4o pricing (as of 2024)
        input_cost_per_token = 0.000005  # $5/1M tokens
        output_cost_per_token = 0.000015  # $15/1M tokens
        
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)