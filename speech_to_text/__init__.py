"""
Speech-to-text module for the Voice AI Agent.

This module provides real-time streaming speech recognition using Google Cloud Speech-to-Text API.
"""

import logging

__version__ = "0.2.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)