"""
Redis Cache Service for Financial Research Chatbot
"""
import json
import logging
from typing import Any, Optional, Dict
from redis.asyncio import Redis
from datetime import datetime

logger = logging.getLogger(__name__)

class CacheService:
    """Redis-based caching service for query responses and user data"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        self.connected = False
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0
        }
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            logger.info("Successfully connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
            logger.info("Disconnected from Redis")
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.connected or not self.redis_client:
            logger.warning("Cache not connected, returning None")
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                self.stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = 3600) -> bool:
        """
        Set value in cache with expiration
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.redis_client:
            logger.warning("Cache not connected, cannot set value")
            return False
        
        try:
            await self.redis_client.setex(key, expire, value)
            self.stats["sets"] += 1
            logger.debug(f"Cached value for key: {key} with TTL: {expire}s")
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error setting cache value: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting cache key: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.connected or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error checking cache key existence: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get time to live for a key
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, None if error
        """
        if not self.connected or not self.redis_client:
            return None
        
        try:
            ttl = await self.redis_client.ttl(key)
            return ttl
            
        except Exception as e:
            logger.error(f"Error getting TTL: {e}")
            return None
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in cache
        
        Args:
            key: Cache key
            amount: Amount to increment (default: 1)
            
        Returns:
            New value or None if error
        """
        if not self.connected or not self.redis_client:
            return None
        
        try:
            result = await self.redis_client.incrby(key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Error incrementing cache value: {e}")
            return None
    
    async def set_hash(self, key: str, data: Dict[str, Any], expire: int = 3600) -> bool:
        """
        Set hash data in cache
        
        Args:
            key: Cache key
            data: Dictionary data to cache
            expire: Expiration time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.redis_client:
            return False
        
        try:
            # Convert data to JSON string
            json_data = json.dumps(data)
            await self.redis_client.setex(key, expire, json_data)
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error setting hash in cache: {e}")
            return False
    
    async def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get hash data from cache
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary data or None if not found
        """
        if not self.connected or not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                self.stats["hits"] += 1
                return json.loads(value)
            else:
                self.stats["misses"] += 1
                return None
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error getting hash from cache: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache service health"""
        try:
            if not self.connected or not self.redis_client:
                return {"status": "disconnected", "error": "Not connected to Redis"}
            
            # Test with ping
            await self.redis_client.ping()
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache service statistics"""
        return {
            "connected": self.connected,
            "stats": self.stats.copy(),
            "last_updated": datetime.now().isoformat()
        }
    
    async def clear_stats(self):
        """Clear cache service statistics"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0
        }
        logger.info("Cache statistics cleared")
    
    async def flush_all(self) -> bool:
        """Flush all cache data (use with caution!)"""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushall()
            logger.warning("All cache data flushed")
            return True
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
