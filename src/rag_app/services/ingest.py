import time
import json
import asyncio
from openai import AsyncOpenAI
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser

from llama_index.core import Document
from llama_index.core.schema import BaseNode

from rag_app.config.logging import logger
from rag_app.embeddings.base import BaseEmbeddingModel
from rag_app.core.vector_client import VectorClient


class IngestionService:
    
    def __init__(self, reader, node_parser, client, embedding_client, vector_client):
        self._pdf_reader: PyMuPDFReader = reader
        self._semantic_node_parser: SemanticSplitterNodeParser = node_parser
        self._llm_client: AsyncOpenAI = client
        self._embedding_client: BaseEmbeddingModel = embedding_client
        self._vector_client: VectorClient = vector_client
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls
        self._failure_threshold = 0.3 # 30% failure threshold

    async def _ingest_file(self, file_path: str) -> list[Document]:
        try:
            logger.info(f"Loading documents from {file_path}")
            documents = await asyncio.get_event_loop().run_in_executor(None, self._pdf_reader.load_data, file_path)
            logger.info(f"Loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            return []
        
    async def _filter_empty_documents(self, documents: list[Document]) -> list[Document]:
        filtered_docs = [doc for doc in documents if doc.get_content().strip()]
        logger.info(f"Filtered documents: {len(filtered_docs)} non-empty out of {len(documents)}")
        return filtered_docs

    async def _preprocess_documents(self, documents: list[Document]) -> list[Document]:
        try:
             logger.info("Starting document preprocessing...")
             preprocessed_docs = await self._filter_empty_documents(documents)
             logger.info("Document preprocessing completed.")
             return preprocessed_docs
        except Exception as e:
            logger.error(f"Failed to preprocess documents: {e}")
            raise e
    
    async def _chunk_documents(self, documents: list[Document]) -> list[BaseNode]:
        try:
            logger.info(f"Chunking {len(documents)} documents into nodes...")
            nodes = await asyncio.get_event_loop().run_in_executor(None, lambda: self._semantic_node_parser.get_nodes_from_documents(documents))
            logger.info(f"Chunked documents into {len(nodes)} nodes.")
            return nodes
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            raise e

    def _build_prompt(self, node: BaseNode) -> str:
        prompt = f"""
        You will be provided with a chunk of text from a document.
        Your task is to extract 5 questions that can be answered using the information in the text.
        you will answer in the following json format
        {{
            "questions": [
                "Question 1",
                "Question 2",
                "Question 3",
                "Question 4",
                "Question 5"
            ]
        }}
        Text: {node.get_content()}       
        """
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        try:
            response = await self._llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts questions from text."},
                    {"role": "user", "content": prompt}
                ],
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_llm_response(self, response: str) -> list[str]:
        try:
            if "```json" in response:
                response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return data.get("questions", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise e
        

    async def _extract_metadata(self, node: BaseNode) -> tuple[BaseNode, bool]:
        async with self._semaphore:
            try:
                prompt = self._build_prompt(node)
                llm_response = await self._call_llm(prompt)
                questions = await asyncio.get_event_loop().run_in_executor(None, lambda: self._parse_llm_response(llm_response))
                node.metadata.update({"questions": questions})
                return node, True
            except Exception as e:
                logger.error(f"Failed to extract metadata: {e}")
                return node, False

    async def _process_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        try:
            logger.info("Extracting metadata for nodes...")
            tasks = [self._extract_metadata(node) for node in nodes]
            processed_nodes = await asyncio.gather(*tasks)
            processed_nodes = [node for node, success in processed_nodes if success]
            failure_count = len(nodes) - len(processed_nodes)
            logger.info("Metadata extraction completed.")

            if failure_count > 0:
                logger.warning(f"Metadata extraction had {failure_count} failures.")

            failure_rate = failure_count / len(nodes)
            if failure_rate > self._failure_threshold:
                logger.error(f"Failure rate {failure_rate:.2%} exceeds threshold of {self._failure_threshold:.2%}. Aborting ingestion.")
                raise Exception("High failure rate in metadata extraction.")
            return processed_nodes
        except Exception as e:
            logger.error(f"Failed to process nodes: {e}")
            raise e
    
    async def _embed_node(self, node: BaseNode) -> tuple[BaseNode, bool]:
        async with self._semaphore:
            try:
                embedding = await self._embedding_client.embed_document(node.get_content())
                node.embedding = embedding
                return node, True
            except Exception as e:
                logger.error(f"Failed to embed node {node.id_}: {e}")
                return node, False

    async def _embed_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        try:
            logger.info("Embedding nodes...")
            tasks = [self._embed_node(node) for node in nodes]
            embedded_nodes = await asyncio.gather(*tasks)
            embedded_nodes = [node for node, success in embedded_nodes if success]
            failure_count = len(nodes) - len(embedded_nodes)
            logger.info("Embedding completed.")

            if failure_count > 0:
                logger.warning(f"Embedding had {failure_count} failures.")

            failure_rate = failure_count / len(nodes)
            if failure_rate > self._failure_threshold:
                logger.error(f"Failure rate {failure_rate:.2%} exceeds threshold of {self._failure_threshold:.2%}. Aborting ingestion.")
                raise Exception("High failure rate in embedding.")
            
            return embedded_nodes
        except Exception as e:
            logger.error(f"Failed to embed nodes: {e}")
            raise e
    
    def _node_to_vector(self, node: BaseNode) -> dict:
        return {
            "id": node.id_,
            "values": node.embedding,
            "metadata": node.metadata
        }
    
    async def _upsert_nodes(self, nodes: list[BaseNode], namespace: str = "") -> None:
        logger.info(f"Upserting {len(nodes)} nodes to vector database...")
        vectors = [self._node_to_vector(node) for node in nodes]
        await self._vector_client.upsert(vectors, namespace=namespace)
        logger.info("Upsert completed.")

    async def _question_to_vector(self, question: str, node: BaseNode, index: int) -> dict:
        async with self._semaphore:
            embedding = await self._embedding_client.embed_document(question)
            return {
                "id": f"{node.id_}_q_{index}",
                "values": embedding,
                "metadata": {
                    "question": question,
                    "node_id": node.id_,
                    "node_content": node.get_content()[:500],  # truncated for metadata
                    "source": node.metadata.get("source", "")
                }
            }

    async def _upsert_questions(self, nodes: list[BaseNode], namespace: str = "questions") -> None:
        logger.info(f"Upserting questions from {len(nodes)} nodes to namespace '{namespace}'...")
        question_tasks = []
        for node in nodes:
            questions = node.metadata.get("questions", [])
            for i, question in enumerate(questions):
                question_tasks.append(self._question_to_vector(question, node, i))
        if question_tasks:
            question_vectors = await asyncio.gather(*question_tasks)
            await self._vector_client.upsert(question_vectors, namespace=namespace)
            logger.info(f"Upserted {len(question_vectors)} question vectors.")
        else:
            logger.info("No questions to upsert.")
    

    async def ingest(self, file_path: str):
        start = time.time()
        documents = await self._ingest_file(file_path)
        documents = await self._preprocess_documents(documents)
        nodes = await self._chunk_documents(documents)
        nodes = await self._process_nodes(nodes)
        nodes = await self._embed_nodes(nodes)
        await self._upsert_nodes(nodes)
        await self._upsert_questions(nodes)
        end = time.time()
        logger.info(f"Ingestion completed in {end - start:.2f} seconds")
        return nodes