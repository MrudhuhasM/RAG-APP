import tiktoken
from functools import lru_cache

@lru_cache()
def get_tokenizer(encoding_name: str = "cl100k_base"):
    """Get a tokenizer for the specified encoding.

    Args:
        encoding_name (str): The name of the encoding to use. Defaults to "cl100k_base".

    Returns:
        tiktoken.Encoding: The tokenizer for the specified encoding.
    """
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str,) -> int:
    if not text:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))