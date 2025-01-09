from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ollama
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rich import print

def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=ollama.OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever: VectorStoreRetriever = vectorstore.as_retriever()
    
    prompt_template = """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
    
    chain = (
        {"context": retriever | str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({"question": "What is the capital of France?"})
