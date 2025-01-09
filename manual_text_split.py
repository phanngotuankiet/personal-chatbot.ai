from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ollama
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rich import print

# Character text splitting
print("Character text splitting")

text = """
The Industrial Revolution marked a major turning point in human history, fundamentally changing how people lived and worked. Beginning in Britain in the late 18th century, this transformation spread across Europe and eventually the world. The shift from manual labor and animal-based production to machine manufacturing and efficient factories created entirely new economic and social systems.
"""

# Manual chunk (text) splitting
print("Manual chunk (text) splitting")
chunks = []
chunk_size = 35 # 35 characters
for i in range(0, len(text), chunk_size):
    chunk: str = text[i:i+chunk_size]
    chunks.append(chunk)
    
documents: list[Document] = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print(documents)