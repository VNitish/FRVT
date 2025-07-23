import time
import hashlib
from typing import Optional, Any, Dict
from collections import OrderedDict
import threading
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache implementation with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                if key in self.cache:
                    del self.cache[key]
                    del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            # Remove expired entries
            self._cleanup_expired()
            
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # Remove oldest if at capacity
                if len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def _cleanup_expired(self) -> None:
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.timestamps[key]
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)

class EmbeddingCache:
    """Specialized cache for face embeddings"""
    
    def __init__(self, max_size: int = 500, ttl: int = 1800):
        self.cache = LRUCache(max_size, ttl)
    
    def _generate_key(self, image_data: bytes) -> str:
        """Generate cache key from image data hash"""
        return hashlib.md5(image_data).hexdigest()
    
    def get_embedding(self, image_data: bytes) -> Optional[np.ndarray]:
        key = self._generate_key(image_data)
        embedding = self.cache.get(key)
        if embedding is not None:
            logger.debug(f"Cache hit for embedding key: {key[:8]}...")
            return embedding
        return None
    
    def put_embedding(self, image_data: bytes, embedding: np.ndarray) -> None:
        key = self._generate_key(image_data)
        self.cache.put(key, embedding.copy())
        logger.debug(f"Cached embedding for key: {key[:8]}...")
    
    def clear(self) -> None:
        self.cache.clear()
        logger.info("Embedding cache cleared")

class ImageProcessingCache:
    """Cache for processed images (cropped, aligned faces)"""
    
    def __init__(self, max_size: int = 200, ttl: int = 900):
        self.cache = LRUCache(max_size, ttl)
    
    def _generate_key(self, image_data: bytes, processing_params: str = "") -> str:
        combined = image_data + processing_params.encode()
        return hashlib.md5(combined).hexdigest()
    
    def get_processed_image(self, image_data: bytes, processing_params: str = "") -> Optional[np.ndarray]:
        key = self._generate_key(image_data, processing_params)
        return self.cache.get(key)
    
    def put_processed_image(self, image_data: bytes, processed_image: np.ndarray, processing_params: str = "") -> None:
        key = self._generate_key(image_data, processing_params)
        self.cache.put(key, processed_image.copy())

# Global cache instances
embedding_cache = EmbeddingCache(max_size=500, ttl=1800)  # 30 minutes
image_cache = ImageProcessingCache(max_size=200, ttl=900)  # 15 minutes

# Collection-specific FAISS index cache
faiss_index_cache = LRUCache(max_size=50, ttl=7200)  # 2 hours
