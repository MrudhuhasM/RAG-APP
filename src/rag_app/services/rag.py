from rag_app.embeddings.base import BaseEmbeddingModel
from rag_app.core.vector_client import VectorClient
from rag_app.core.cache_client import CacheClient
from rag_app.config.logging import logger
from rag_app.config.settings import settings
from rag_app.llm.base import BaseLLMModel
import time
import uuid
import asyncio
from typing import AsyncGenerator, Optional
from sentence_transformers import CrossEncoder
from typing import AsyncGenerator, Optional
from rag_app.utils.tokenizer import count_tokens, get_tokenizer
from rag_app.services.router import QueryRouter
from rag_app.llm import get_llm_model_by_provider

class RagService:
    def __init__(self,
                 embed_model: BaseEmbeddingModel,
                 vector_client: VectorClient,
                 llm_model: BaseLLMModel,
                 encoder_model: CrossEncoder,
                 cache_client: CacheClient,
                 router: Optional[QueryRouter] = None):
        
        self._embed_model: BaseEmbeddingModel = embed_model
        self._vector_client: VectorClient = vector_client
        self._llm_model: BaseLLMModel = llm_model
        self._reranker: CrossEncoder = encoder_model
        self._cache_client: CacheClient = cache_client
        self._router: Optional[QueryRouter] = router
        self._available_models: dict[str, BaseLLMModel] = {}  # Cache for dynamically loaded models

    async def _get_llm_for_query(self, query: str) -> BaseLLMModel:
        """
        Determine which LLM model to use for the given query.
        If router is available, route the query. Otherwise, use default LLM.
        
        Args:
            query (str): The user query.
            
        Returns:
            BaseLLMModel: The selected LLM model instance.
        """
        if self._router is None:
            logger.debug("No router available, using default LLM model")
            return self._llm_model
        
        try:
            logger.info("Starting query routing to select optimal LLM model")
            # Route the query to determine the provider
            provider = await self._router.route_query(query)
            logger.info(f"✓ Router selected provider: '{provider}'")
            
            # Return cached model if available
            if provider in self._available_models:
                logger.debug(f"Using cached LLM model for provider: {provider}")
                return self._available_models[provider]
            
            # Load and cache the model
            logger.info(f"Loading new LLM model for provider: {provider}")
            model = get_llm_model_by_provider(provider)
            self._available_models[provider] = model
            logger.info(f"✓ Successfully loaded and cached model: {model.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"✗ Query routing failed with error: {type(e).__name__}: {e}")
            logger.warning(f"Falling back to default LLM model: {self._llm_model.model_name}")
            return self._llm_model

    async def rerank(self, query: str, contexts: list[dict], top_k: Optional[int] = None) -> list[dict]:
        """
        Reranks the retrieved contexts based on their relevance to the query.

        Args:
            query (str): The user query.
            contexts (list[dict]): The list of context documents to rerank.
            top_k (int, optional): The number of top contexts to return. Defaults to settings value.

        Returns:
            list[dict]: The reranked list of context documents.
        """
        if top_k is None:
            top_k = settings.rerank_top_k
        try:
            pairs = [(query, context['content']) for context in contexts]
            scores = await asyncio.to_thread(self._reranker.predict, pairs)
            for i, context in enumerate(contexts):
                context['rerank_score'] = float(scores[i])  # Convert to Python float
            ranked_contexts = sorted(contexts, key=lambda x: x['rerank_score'], reverse=True)
            return ranked_contexts[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return contexts[:top_k]

    async def _check_semantic_cache(self, query_emb: list[float]) -> list[dict]:
        try:
            logger.debug("Checking semantic cache for relevant documents.")
            results = await self._vector_client.query(
                vector=query_emb,
                top_k=1,
                include_metadata=True,
                namespace=settings.pinecone.semantic_cache_namespace
            )

            if not results['matches']:
                logger.debug("No matches found in semantic cache.")
                return []

            top_match = results['matches'][0]
            if top_match['score'] >= settings.semantic_threshold:
                logger.info(f"Semantic cache hit with score: {top_match['score']}")
                redis_key = top_match['metadata'].get('redis_key')
                if not redis_key:
                    logger.warning("No redis_key found in semantic cache metadata.")
                    return []
                
                cached_data = await self._cache_client.get(redis_key)
                if cached_data:
                    logger.info("Retrieved data from Redis cache based on semantic cache hit.")
                    return cached_data
                else:
                    logger.warning("No data found in Redis for the given redis_key.")
                    return []
                
            logger.debug("Semantic cache miss based on threshold.")
            return []       

        except Exception as e:
            logger.error(f"Semantic cache check failed: {e}")
            return []
        
    
    async def _add_to_semantic_cache(self, query_emb: list[float], redis_key: str) -> None:

        try:
            vector_id = str(uuid.uuid4())
            await self._vector_client.upsert(
                vectors=[{
                    'id': vector_id,
                    'values': query_emb,
                    'metadata': {'redis_key': redis_key}
                }],
                namespace=settings.pinecone.semantic_cache_namespace
            )
            logger.info(f"Added new entry to semantic cache with vector ID: {vector_id}")
        except Exception as e:
            logger.error(f"Adding to semantic cache failed: {e}")
            return



    async def query_rewriter(self, query: str) -> str:
        """
        Rewrites the user query to be more specific using the LLM model.

        Args:
            query (str): The original user query.
        Returns:
            str: The rewritten query.
        """
        try:
            prompt = f"""
            You are a query rewriter that takes a user query and rewrites it to be more specific.
            Make sure your rewritten query captures the intent of the original query.
            without any confusion or ambiguity.
            This RAG system answers questions on Complete Works of Mahatma Gandhi.
            Give only the rewritten query as output.No other text.or explanation.
            If the query is already specific, return it as is.
            Do not in any circumstance give an empty response.
            Original Query: {query}
            Rewritten Query:"""
            messages = [
                {"role": "system", "content": "You are a helpful assistant that rewrites user queries to be more specific."},
                {"role": "user", "content": prompt}
            ]
            response = await self._llm_model.generate_response(
                messages=messages,
            )

            return response
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query

    async def retrieve_documents(self, query_emb: list[float], top_k: Optional[int] = None, namespace: str = "") -> list[dict]:
        """
        Retrieves documents from the vector store based on the query embedding.

        Args:
            query_emb (list[float]): The embedding vector of the user query.
            top_k (int, optional): The number of top documents to retrieve. Defaults to settings value.
            namespace (str, optional): The namespace to query within. Defaults to "".

        Returns:
            list[dict]: The list of retrieved documents.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k
        try:
            results = await self._vector_client.query(
                vector=query_emb,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            return results['matches']
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def format_results(self, results: list[dict]):
        """
        Formats the raw results from the vector store into a structured format.

        Args:
            results (list[dict]): The raw results from the vector store.

        Returns:
            list[dict]: The formatted results.
        """
        if not results:
            return []
        try:
            final_results = []
            for result in results:
                final_results.append({"content": result['metadata'].get('node_content', ''),
                                      "score": float(result['score']),  # Convert to Python float
                                      "source": result['metadata'].get('source', 'unknown')})
            return final_results
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return []
    
    def remove_duplicates(self, results: list[dict]) -> list[dict]:
        """
        Removes duplicate results based on content.

        Args:
            results (list[dict]): The list of results to filter.

        Returns:
            list[dict]: The list of unique results.
        """
        if not results:
            return []
        try:
            seen = set()
            unique_results = []
            for result in results:
                if result['content'] not in seen:
                    seen.add(result['content'])
                    unique_results.append(result)
            return unique_results
        except Exception as e:
            logger.error(f"Duplicate removal failed: {e}")
            return results
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncates the text to the specified number of tokens.

        Args:
            text (str): The text to truncate.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The truncated text.
        """
        if count_tokens(text) <= max_tokens:
            return text
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    
    def build_context_with_limit(self, contexts: list[dict], max_tokens: Optional[int] = None) -> str:
        """
        Builds the context string from contexts, limiting to max_tokens.

        Args:
            contexts (list[dict]): The list of context documents.
            max_tokens (int): The maximum number of tokens for the context. Defaults to settings value.

        Returns:
            str: The context string within the token limit.
        """
        if max_tokens is None:
            max_tokens = settings.context_max_tokens
        context_parts = []
        current_tokens = 0
        for ctx in contexts:
            source_part = f"Source: {ctx['source']}\nContent: "
            source_tokens = count_tokens(source_part)
            content = ctx['content']
            content_tokens = count_tokens(content)
            total_part_tokens = source_tokens + content_tokens
            if current_tokens + total_part_tokens <= max_tokens:
                part = source_part + content
                context_parts.append(part)
                current_tokens += total_part_tokens
            else:
                remaining_tokens = max_tokens - current_tokens - source_tokens
                if remaining_tokens > 0:
                    truncated_content = self.truncate_text(content, remaining_tokens)
                    part = source_part + truncated_content
                    context_parts.append(part)
                break
        return "\n\n".join(context_parts)
    
    async def generate_answer(self, query: str, contexts: list[dict], llm_model: BaseLLMModel) -> AsyncGenerator[str, None]:
        """
        Generates a final answer based on the top contexts using the provided LLM model.

        Args:
            query (str): The user query.
            contexts (list[dict]): The list of context documents.
            llm_model (BaseLLMModel): The LLM model to use for generation.

        Returns:
            str: The generated answer.
        """

        if not contexts:
            logger.warning("No contexts available to generate an answer.")
            yield "No contexts available to generate an answer."
            return
        if not query:
            yield "Please provide a valid query."
            return
        
        try:
            logger.info(f"Using LLM model: {llm_model.model_name}")
            
            context_texts = self.build_context_with_limit(contexts)
            prompt = f"""
            You are an advanced AI assistant tasked with answering questions based strictly on the provided contexts.
            Your response should be concise, accurate, and directly supported by the contexts.
            If the answer cannot be found in the contexts, respond with 'I do not have relevant information to answer that question.'.
            Use the contexts to construct a clear and well-structured answer.
            
            Contexts:
            {context_texts}
            
            Question: {query}
            Answer:
            """
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided contexts."},
                {"role": "user", "content": prompt}
            ]
            
            async for chunk in llm_model.stream_completion(messages=messages):
                yield chunk

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            yield "An error occurred while generating the answer."


    async def answer_query(self, query: str) -> AsyncGenerator[dict, None]:
        """
        Answers a user query using the RAG approach.

        Args:
            query (str): The user query.

        Returns:
            dict: The answer and sources.
        """

        if not query:
            logger.warning("Received empty query.")
            yield {"type": "error", "data": "Please provide a valid query."}

        input_tokens = 0 
        output_tokens = 0
        model_name = "unknown"  # Will be set after routing

        try:
            input_tokens = count_tokens(query)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
        
        try:
            cache_key = f"rag_answer:{query}"
            result_cache = await self._cache_client.get(cache_key)
            if result_cache:
                logger.info("Cache hit for query.")
                yield {"type": "status", "data": "Fetching answer from cache."}
                yield {"type": "token", "data": result_cache['answer']}
                yield {"type": "sources", "data": result_cache['sources']}
                return
            logger.info("Cache miss for query.")
            start_time = time.time()
            logger.info(f"Starting RAG retrieval for query: {query}")
            yield {"type": "status", "data": "Rewriting query."}
            rewritten_query = await self.query_rewriter(query)
            logger.info(f"Rewritten query: {rewritten_query}")

            logger.info("Generating embedding for rewritten query.")
            yield {"type": "status", "data": "Generating query embedding."}
            embedding_cache_key = f"embedding:{rewritten_query}"
            query_embedding = await self._cache_client.get(embedding_cache_key)
            if not query_embedding:
                logger.info("Embedding cache miss, generating new embedding.")
                query_embedding = await self._embed_model.embed_document(rewritten_query)
                await self._cache_client.set(embedding_cache_key, query_embedding, is_embedding=True)

            semantic_cached_results = await self._check_semantic_cache(query_embedding)
            if semantic_cached_results:
                await self._cache_client.set(cache_key, {"answer": semantic_cached_results['answer'], "sources": semantic_cached_results['sources']})

                logger.info("Returning answer from semantic cache.")
                yield {"type": "token", "data": semantic_cached_results['answer']}
                yield {"type": "sources", "data": semantic_cached_results['sources']}
                return

            logger.info("Retrieving relevant documents from vector store.")
            yield {"type": "status", "data": "Retrieving relevant documents."}

            doc_task = self.retrieve_documents(query_embedding)
            question_task = self.retrieve_documents(
                query_embedding, 
                top_k=settings.retrieval_questions_top_k, 
                namespace=settings.pinecone.questions_namespace
            )

            results, question_results = await asyncio.gather(doc_task, question_task)
            logger.info(f"Retrieved {len(results)} relevant documents and {len(question_results)} relevant questions.")
            
            total_results = results + question_results
            final_results = self.format_results(total_results)            
            
            # Remove duplicates based on content
            final_results = self.remove_duplicates(final_results)
            
            logger.info("Reranking the retrieved results.")
            yield {"type": "status", "data": "Reranking retrieved documents."}
            final_results = await self.rerank(rewritten_query, final_results)

            logger.info("Generating final answer from top contexts.")
            yield {"type": "status", "data": "Generating final answer."}            
            full_answer = []
            
            # Get the LLM model that will be used (for cost tracking) - ONLY ONCE!
            selected_llm = await self._get_llm_for_query(rewritten_query)
            model_name = selected_llm.model_name
            logger.info(f"Selected model for answer generation: {model_name}")
            
            async for chunk in self.generate_answer(rewritten_query, final_results, selected_llm):
                full_answer.append(chunk)
                yield {"type": "token", "data": chunk}

            answer_text = ''.join(full_answer)

            error_responses = [
                "I do not have relevant information to answer that question.",
                "No contexts available to generate an answer.",
                "An error occurred while generating the answer.",
                "Please provide a valid query.",
                "",
                " "
            ]
            # Cache the final answer and sources
            if answer_text not in error_responses:
                await self._cache_client.set(cache_key, {"answer": answer_text, "sources": final_results})
                await self._add_to_semantic_cache(query_embedding, cache_key)
                logger.info("Final answer and sources cached.")
                
            yield {"type": "sources", "data": final_results}
            elapsed_time = time.time() - start_time
            logger.info(f"RAG retrieval completed in {elapsed_time:.2f} seconds.")
            
            output_tokens = count_tokens(answer_text)
            model_costs = settings.MODEL_COSTS.get(model_name)

            if not model_costs:
                logger.warning(f"Model costs not defined for model: {model_name}")

            input_cost = (input_tokens / 1000) * model_costs['input'] if model_costs else 0
            output_cost = (output_tokens / 1000) * model_costs['output'] if model_costs else 0
            total_cost = input_cost + output_cost

            logger.info(f"Request Processed: Input tokens {input_tokens}, Output tokens {output_tokens}, Input cost {input_cost}, Output cost {output_cost}, Total cost {total_cost}, Model {model_name}")
        except Exception as e:
            logger.error(f"Error in answering query: {e}")
            yield {"type": "error", "data": "An error occurred while processing your query."}
