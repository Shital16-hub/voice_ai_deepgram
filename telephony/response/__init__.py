"""
Response processing components for telephony integration.
"""
from .echo_detection import EchoDetection
from .knowledge_base_processor import KnowledgeBaseProcessor

__all__ = [
    'EchoDetection',
    'KnowledgeBaseProcessor'
]