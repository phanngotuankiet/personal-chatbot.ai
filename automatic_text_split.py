from typing import List
from langchain_core.documents.base import Document
from rich import print

# Automatic text splitting
from langchain.text_splitter import CharacterTextSplitter
print("Automatic text splitting")

text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=0, separator="")
text = """
The Industrial Revolution marked a major turning point in human history, fundamentally changing how people lived and worked. Beginning in Britain in the late 18th century, this transformation spread across Europe and eventually the world. The shift from manual labor and animal-based production to machine manufacturing and efficient factories created entirely new economic and social systems.
"""

document: List[Document] = text_splitter.create_documents([text])
print(document)