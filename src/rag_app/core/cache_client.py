from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError
from rag_app.config.logging import logger
import json
import asyncio
from typing import Any, Optional


class CacheClient:
    def __init__(self, connection_pool: ConnectionPool, ttl_seconds: Optional[int] = None, embedding_ttl_seconds: Optional[int] = None):
        self.redis = Redis(connection_pool=connection_pool)
        self.ttl_seconds = ttl_seconds
        self.embedding_ttl_seconds = embedding_ttl_seconds

    @classmethod
    async def create(cls, host: str, port: int, db: int, ttl_seconds: Optional[int] = None, embedding_ttl_seconds: Optional[int] = None, max_retries: int = 10) -> "CacheClient":
        """
        Create a CacheClient with retry logic for connecting to Redis.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl_seconds: Default TTL for cache entries
            embedding_ttl_seconds: TTL for embedding cache entries
            max_retries: Maximum number of connection attempts
            
        Returns:
            CacheClient instance
            
        Raises:
            RedisError: If connection fails after all retries
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                pool = ConnectionPool(
                    host=host, 
                    port=port, 
                    db=db,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=10
                )
                client = cls(connection_pool=pool, ttl_seconds=ttl_seconds, embedding_ttl_seconds=embedding_ttl_seconds)
                
                # Test connection with ping
                await client.redis.ping()
                logger.info(f"Connected to Redis cache successfully at {host}:{port}")
                return client
                
            except (RedisError, ConnectionError, OSError) as e:
                last_error = e
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                logger.warning(
                    f"Failed to connect to Redis (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts: {last_error}", exc_info=True)
                    raise RedisError(f"Could not connect to Redis at {host}:{port} after {max_retries} attempts") from last_error

    async def set(self, key: str, value: Any, is_embedding: bool = False) -> None:
        try:
            ttl = self.embedding_ttl_seconds if is_embedding else self.ttl_seconds
            serialized_value = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized_value)
            else:
                await self.redis.set(key, serialized_value)
        except RedisError as e:
            logger.error(f"Failed to set key {key} in Redis: {e}", exc_info=True)
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            serialized_value = await self.redis.get(key)
            if serialized_value is None:
                return None
            return json.loads(serialized_value.decode('utf-8'))
        except RedisError as e:
            logger.error(f"Failed to get key {key} from Redis: {e}", exc_info=True)
            raise

    async def delete(self, key: str) -> None:
        try:
            await self.redis.delete(key)
        except RedisError as e:
            logger.error(f"Failed to delete key {key} from Redis: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        try:
            await self.redis.close()
            logger.info("Redis connection closed successfully")
        except RedisError as e:
            logger.error(f"Failed to close Redis connection: {e}", exc_info=True)
            raise

