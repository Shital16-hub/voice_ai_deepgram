# telephony/__init__.py - Option 1: Minimal imports to avoid circular dependencies

"""
Telephony integration package for Voice AI Agent.
"""

# Only import what's absolutely necessary to avoid circular dependencies
# Remove imports that cause the circular dependency chain

__all__ = [
    # 'TwilioHandler',  # Commented out to break the import chain
    # 'AudioProcessor',  # Commented out to break the import chain  
    # 'WebSocketHandler',  # Commented out to break the import chain
    'CallManager'
]

# Import only CallManager which doesn't have problematic dependencies
from telephony.call_manager import CallManager