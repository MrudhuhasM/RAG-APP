from rag_app.embeddings.base import BaseEmbeddingModel
from rag_app.core.vector_client import VectorClient
from rag_app.config.logging import logger
from rag_app.config.settings import settings
from rag_app.llm.base import BaseLLMModel
import time
import asyncio
from sentence_transformers import CrossEncoder

class RagService:
    def __init__(self,
                 embed_model: BaseEmbeddingModel,
                 vector_client: VectorClient,
                 llm_model: BaseLLMModel,
                 encoder_model: CrossEncoder):
        
        self._embed_model: BaseEmbeddingModel = embed_model
        self._vector_client: VectorClient = vector_client
        self._llm_model: BaseLLMModel = llm_model
        self._reranker: CrossEncoder = encoder_model

    async def rerank(self,query: str, contexts: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reranks the retrieved contexts based on their relevance to the query.

        Args:
            query (str): The user query.
            contexts (list[dict]): The list of context documents to rerank.
            top_k (int, optional): The number of top contexts to return. Defaults to 5.

        Returns:
            list[dict]: The reranked list of context documents.
        """
        try:
            pairs = [(query, context['content']) for context in contexts]
            scores = await asyncio.to_thread(self._reranker.predict, pairs)
            for i, context in enumerate(contexts):
                context['rerank_score'] = scores[i]
            ranked_contexts = sorted(contexts, key=lambda x: x['rerank_score'], reverse=True)
            return ranked_contexts[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return contexts[:top_k]

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
            Original Query: {query}
            Rewritten Query:"""
            messages = [
                {"role": "system", "content": "You are a helpful assistant that rewrites user queries to be more specific."},
                {"role": "user", "content": prompt}
            ]
            response = await self._llm_model.generate_completion(
                messages=messages,
                model=settings.openai.completion_model,
            )

            return response
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query

    async def retrieve_documents(self, query_emb: list[float], top_k: int = 10, namespace: str = "") -> list[dict]:
        """
        Retrieves documents from the vector store based on the query embedding.

        Args:
            query_emb (list[float]): The embedding vector of the user query.
            top_k (int, optional): The number of top documents to retrieve. Defaults to 10.
            namespace (str, optional): The namespace to query within. Defaults to "".

        Returns:
            list[dict]: The list of retrieved documents.
        """
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
                                      "score": result['score'],
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
    
    async def generate_answer(self, query: str, contexts: list[dict]) -> str:
        """
        Generates a final answer based on the top contexts.

        Args:
            query (str): The user query.
            contexts (list[dict]): The list of context documents.

        Returns:
            str: The generated answer.
        """

        if not contexts:
            return "No contexts available to generate an answer."
        if not query:
            return "Please provide a valid query."
        
        try:        
            context_texts = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['content']}" for ctx in contexts])
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
            response = await self._llm_model.generate_completion(
                messages=messages,
                model=settings.openai.completion_model,
            )

            return response
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "An error occurred while generating the answer."


    async def answer_query(self, query: str) -> dict:
        """
        Answers a user query using the RAG approach.

        Args:
            query (str): The user query.

        Returns:
            dict: The answer and sources.
        """

        if not query:
            return {"answer": "Please provide a valid query.", "sources": []}
        
        try:
            start_time = time.time()
            logger.info(f"Starting RAG retrieval for query: {query}")
            rewritten_query = await self.query_rewriter(query)
            logger.info(f"Rewritten query: {rewritten_query}")

            logger.info("Generating embedding for rewritten query.")
            query_embedding = await self._embed_model.embed_document(rewritten_query)

            logger.info("Retrieving documents and questions from vector store.")
            results = await self.retrieve_documents(query_embedding, top_k=10)
            question_results = await self.retrieve_documents(query_embedding, top_k=20, namespace=settings.pinecone.questions_namespace )
            logger.info(f"Retrieved {len(results)} relevant documents and {len(question_results)} relevant questions.")
            
            total_results = results + question_results
            final_results = self.format_results(total_results)            
            
            # Remove duplicates based on content
            final_results = self.remove_duplicates(final_results)
            
            logger.info("Reranking the retrieved results.")
            final_results = await self.rerank(rewritten_query, final_results, top_k=5)

            logger.info("Generating final answer from top contexts.")
            answer = await self.generate_answer(rewritten_query, final_results)
            elapsed_time = time.time() - start_time
            logger.info(f"RAG retrieval completed in {elapsed_time:.2f} seconds.")
            return {"answer": answer, "sources": final_results}
        except Exception as e:
            logger.error(f"Error in answering query: {e}")
            return {"answer": "An error occurred while processing your query.", "sources": []}
