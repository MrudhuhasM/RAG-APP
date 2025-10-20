import time
import json
from openai import OpenAI
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
        self._llm_client: OpenAI = client
        self._embedding_client: BaseEmbeddingModel = embedding_client
        self._vector_client: VectorClient = vector_client

    def _ingest_file(self, file_path: str) -> list[Document]:
        logger.info(f"Loading documents from {file_path}")
        documents = self._pdf_reader.load_data(file_path)
        logger.info(f"Loaded {len(documents)} documents.")
        return documents
    
    def _preprocess_documents(self, documents: list[Document]) -> list[Document]:
        logger.info("Preprocessing documents...")
        preprocessed_docs = [ doc for doc in documents if doc.get_text().strip() ]
        logger.info(f"Preprocessed documents. {len(documents)} -> {len(preprocessed_docs)}")
        return preprocessed_docs
    
    def _chunk_documents(self, documents: list[Document]) -> list[BaseNode]:
        logger.info(f"Chunking {len(documents)} documents into nodes...")
        nodes = self._semantic_node_parser.get_nodes_from_documents(documents)
        logger.info(f"Chunked documents into {len(nodes)} nodes.")
        return nodes
    
    def _build_prompt(self, node: BaseNode) -> str:
        prompt = f"""
        You will be provided with a chunk of text from a document.
        Your task is to extract 5 questions that can be answered using the information in the text.
        you will anwer in follwoing json format
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
    
    def _call_llm(self, prompt: str) -> str:
        response = self._llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts questions from text."},
                {"role": "user", "content": prompt}
            ],
        )
        content = response.choices[0].message.content
        return content if content else ""

    def _parse_llm_response(self, response: str) -> list[str]:
        try:
            if "```json" in response:
                response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return data.get("questions", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []      
        

    def _extract_metadata(self, node: BaseNode) -> BaseNode:
        prompt = self._build_prompt(node)
        llm_response = self._call_llm(prompt)
        questions = self._parse_llm_response(llm_response)
        node.metadata.update({"questions": questions})
        return node       
    
    def _process_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        logger.info("Extracting metadata for nodes...")
        processed_nodes = [self._extract_metadata(node) for node in nodes]
        logger.info("Metadata extraction completed.")
        return processed_nodes
    
    def _embed_node(self, node: BaseNode) -> BaseNode:
        embedding = self._embedding_client.embed_document(node.get_content())
        node.embedding = embedding
        return node
    
    def _embed_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        logger.info("Embedding nodes...")
        embedded_nodes = [self._embed_node(node) for node in nodes]
        logger.info("Embedding completed.")
        return embedded_nodes
    
    def _node_to_vector(self, node: BaseNode) -> dict:
        return {
            "id": node.id_,
            "values": node.embedding,
            "metadata": node.metadata
        }
    
    def _upsert_nodes(self, nodes: list[BaseNode], namespace: str = "") -> None:
        logger.info(f"Upserting {len(nodes)} nodes to vector database...")
        vectors = [self._node_to_vector(node) for node in nodes]
        self._vector_client.upsert(vectors, namespace=namespace)
        logger.info("Upsert completed.")
    
    def _question_to_vector(self, question: str, node: BaseNode, index: int) -> dict:
        embedding = self._embedding_client.embed_document(question)
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
    
    def _upsert_questions(self, nodes: list[BaseNode], namespace: str = "questions") -> None:
        logger.info(f"Upserting questions from {len(nodes)} nodes to namespace '{namespace}'...")
        question_vectors = []
        for node in nodes:
            questions = node.metadata.get("questions", [])
            for i, question in enumerate(questions):
                vector = self._question_to_vector(question, node, i)
                question_vectors.append(vector)
        if question_vectors:
            self._vector_client.upsert(question_vectors, namespace=namespace)
            logger.info(f"Upserted {len(question_vectors)} question vectors.")
        else:
            logger.info("No questions to upsert.")
    

    def ingest(self, file_path: str):
        start = time.time()
        documents = self._ingest_file(file_path)
        documents = self._preprocess_documents(documents)
        nodes = self._chunk_documents(documents[20:24])
        nodes = self._process_nodes(nodes)
        nodes = self._embed_nodes(nodes)
        self._upsert_nodes(nodes)
        self._upsert_questions(nodes)
        end = time.time()
        logger.info(f"Ingestion completed in {end - start:.2f} seconds")
        return nodes