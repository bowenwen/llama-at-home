import os

from langchain.text_splitter import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    TokenTextSplitter,
)


@staticmethod
def get_secrets(key_name):
    _key_path = f"secrets/{key_name}.key"
    if os.path.exists(_key_path):
        _key_file = open(_key_path, "r", encoding="utf-8")
        _key_value = _key_file.read()
    else:
        _key_value = None
    return _key_value


@staticmethod
def get_default_text_splitter(method) -> TextSplitter:
    # Note on different chunking strategies https://www.pinecone.io/learn/chunking-strategies/
    # Note that RecursiveCharacterTextSplitter can currently enter infinite loop:
    # see https://github.com/hwchase17/langchain/issues/1663
    method = method.lower()
    if method == "character":
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    elif method == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    elif method == "nltk":
        text_splitter = NLTKTextSplitter(chunk_size=1000)
    elif method == "spacy":
        text_splitter = SpacyTextSplitter(chunk_size=1000)
    elif method == "tiktoken":
        text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
    else:
        raise ValueError(f"argument method {method} is not supported")
    return text_splitter
