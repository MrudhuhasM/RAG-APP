import asyncio
import logging
from typing import List, Dict, Any, Optional
from pinecone import PineconeAsyncio, ServerlessSpec
from pinecone.exceptions import PineconeException

logger = logging.getLogger(__name__)


class VectorClient:
    """
    A robust client for interacting with Pinecone vector database.

    Supports operations like upserting vectors, querying for similar vectors,
    and managing namespaces for data isolation.
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize the VectorClient.

        Args:
            api_key: Pinecone API key
            index_name: Name of the index to use/create
            dimension: Vector dimension (must match embedding model)
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            region: Cloud region
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        self.pc = PineconeAsyncio(api_key=api_key)
        self.spec = ServerlessSpec(cloud=cloud, region=region)
        self.index: Optional[Any] = None  # Will be set in create

    @classmethod
    async def create(
        cls,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Asynchronously create and initialize the VectorClient.

        Args:
            Same as __init__

        Returns:
            Initialized VectorClient instance
        """
        instance = cls(api_key, environment, index_name, dimension, metric, cloud, region)
        instance.index = await instance._ensure_index_async()
        logger.info(f"VectorClient initialized for index '{index_name}'")
        return instance

    async def _ensure_index_async(self):
        """Ensure the index exists, create if it doesn't."""
        try:
            if not await self.pc.has_index(self.index_name):
                logger.info(f"Creating index '{self.index_name}' with dimension {self.dimension}")
                await self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.spec,
                    tags={"environment": self.environment}
                )
                logger.info(f"Index '{self.index_name}' created successfully")
            else:
                logger.info(f"Index '{self.index_name}' already exists")
            return self.pc.IndexAsyncio(self.index_name)
        except PineconeException as e:
            logger.error(f"Failed to create/ensure index: {e}")
            raise

    async def upsert(
        self,
        vectors: List[Any],
        namespace: str = "",
        batch_size: int = 100
    ) -> None:
        """
        Upsert vectors into the index.

        Args:
            vectors: List of vector dicts with 'id', 'values', and optional 'metadata'
            namespace: Namespace for isolation
            batch_size: Number of vectors to upsert in each batch
        """
        if not vectors:
            logger.warning("No vectors provided for upsert")
            return

        try:
            # Split into batches to avoid payload size limits
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                await self.index.upsert(vectors=batch, namespace=namespace)
                logger.debug(f"Upserted batch of {len(batch)} vectors to namespace '{namespace}'")

            logger.info(f"Successfully upserted {len(vectors)} vectors to namespace '{namespace}'")
        except PineconeException as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        include_metadata: bool = True,
        include_values: bool = False,
        filter: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Query for similar vectors.

        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Namespace to search in
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results
            filter: Metadata filter dict

        Returns:
            Query results dict
        """
        try:
            results = await self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
                include_values=include_values,
                filter=filter
            )
            logger.debug(f"Query executed in namespace '{namespace}'")
            return results
        except PineconeException as e:
            logger.error(f"Failed to query vectors: {e}")
            raise

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: str = "",
        delete_all: bool = False
    ) -> None:
        """
        Delete vectors from the index.

        Args:
            ids: List of vector IDs to delete
            namespace: Namespace to delete from
            delete_all: Whether to delete all vectors in namespace
        """
        try:
            if delete_all:
                await self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted all vectors in namespace '{namespace}'")
            elif ids:
                await self.index.delete(ids=ids, namespace=namespace)
                logger.info(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
            else:
                logger.warning("No ids or delete_all specified for delete operation")
        except PineconeException as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

    async def get_index_stats(self) -> Any:
        """
        Get statistics about the index.

        Returns:
            Index stats dict
        """
        try:
            stats = await self.index.describe_index_stats()
            logger.debug(f"Retrieved index stats: {stats}")
            return stats
        except PineconeException as e:
            logger.error(f"Failed to get index stats: {e}")
            raise

    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the index.

        Returns:
            List of namespace names
        """
        try:
            stats = await self.get_index_stats()
            namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            logger.debug(f"Found namespaces: {namespaces}")
            return namespaces
        except PineconeException as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise