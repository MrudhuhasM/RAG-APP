from rag_app.config.logging import logger
from rag_app.config.settings import settings
from rag_app.llm.base import BaseLLMModel


MODEL_MAP = {
    "simple": "local",
    "complex": "gemini"
}

CLASSIFICATION_PROMPT = """
Analyze the user's query and classify its complexity.

Classification Rules:
- SIMPLE: A straightforward factual question, a basic lookup, or a simple definition. Examples: "Who is Gandhi?", "When was he born?", "What is ahimsa?"
- COMPLEX: Requires multi-step reasoning, comparison, analysis, or synthesis of multiple concepts. Examples: "Compare Gandhi's philosophy with...", "Analyze the impact of...", "How did X influence Y?"

User Query: "{query}"

Respond with ONLY ONE WORD: either "SIMPLE" or "COMPLEX"

Classification:"""

class QueryRouter:
    def __init__(self, llm_model: BaseLLMModel):
        self.llm_model = llm_model
        logger.info(f"QueryRouter initialized with LLM model: {llm_model.__class__.__name__}")

    async def _classify_query(self, query: str) -> bool:
        """
        Classify the query as simple or complex.
        
        Args:
            query: The user query to classify
            
        Returns:
            bool: True if complex, False if simple
        """
        try:
            logger.debug(f"Classifying query: {query[:100]}...")
            prompt = CLASSIFICATION_PROMPT.format(query=query)
            messages = [
                {"role": "system", "content": "You are a query complexity classifier. Respond with only SIMPLE or COMPLEX."},
                {"role": "user", "content": prompt}
            ]
            
            logger.debug("Calling LLM for query classification")
            response = await self.llm_model.generate_response(messages, temperature=0.3)
            
            # Parse the text response
            response_text = response.strip().upper()
            logger.debug(f"Raw classification response: {response_text}")
            
            # Check if response contains "COMPLEX"
            is_complex = "COMPLEX" in response_text
            
            logger.info(f"Query classified as {'COMPLEX' if is_complex else 'SIMPLE'}")
            return is_complex
            
        except Exception as e:
            logger.error(f"Query classification failed: {type(e).__name__}: {e}", exc_info=True)
            logger.warning("Defaulting to 'SIMPLE' classification due to error")
            # Default to simple on error (safer, uses cheaper model)
            return False
    
    async def route_query(self, query: str) -> str:
        """
        Route the query to the appropriate LLM provider based on complexity.
        
        Args:
            query: The user query to route
            
        Returns:
            str: The provider name ("local" or "gemini")
        """
        logger.debug(f"Routing query: {query[:100]}...")
        is_complex = await self._classify_query(query)
        
        if is_complex:
            provider = MODEL_MAP["complex"]
            logger.info(f"Query is COMPLEX → routing to '{provider}' provider")
        else:
            provider = MODEL_MAP["simple"]
            logger.info(f"Query is SIMPLE → routing to '{provider}' provider")

        return provider
