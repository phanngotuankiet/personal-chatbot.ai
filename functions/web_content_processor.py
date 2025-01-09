# Text splitter để chia nhỏ nội dung thành các chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vector database để lưu trữ và tìm kiếm các vector
from langchain_chroma import Chroma

# embedding model để vector hoá text
from langchain_huggingface import HuggingFaceEmbeddings

# document để lưu trữ nội dung của mỗi chunk
from langchain_core.documents.base import Document

# LLM để xử lý nội dung
from langchain_ollama import OllamaLLM

from typing import Dict, List, Tuple
import os
from datetime import datetime

from tools.web_scraper_tool import WebContentResponse
from types_api.types_api import ProcessWebContentsResponse

from numpy import dot
from numpy.linalg import norm

class WebContentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.llm = OllamaLLM(
            model="qwen2",
            verbose=True
        )
        
        self.similarity_threshold = 0.5
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.persist_dir = f"./db/chroma_{timestamp}"
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Tính cosine similarity giữa 2 đoạn text"""
        # Vector hóa cả 2 text
        embedding1: List[float] = self.embeddings.embed_query(text1)
        embedding2: List[float] = self.embeddings.embed_query(text2)
        
        # Tính cosine similarity
        cos_sim = dot(embedding1, embedding2)/(norm(embedding1)*norm(embedding2))
        return float(cos_sim)
        
    async def filter_chunks_by_similarity(
        self, 
        chunks: list[str], 
        query: str
    ) -> list[str]:
        """Filter chunks dựa trên similarity với query"""
        relevant_chunks = []
        
        # Vector hóa query một lần
        query_embedding = self.embeddings.embed_query(query)
        
        # Vector hóa tất cả chunks
        chunk_embeddings = [
            self.embeddings.embed_query(chunk) 
            for chunk in chunks
        ]
        
        # Tính similarity và filter
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = dot(query_embedding, chunk_embedding)/(
                norm(query_embedding)*norm(chunk_embedding)
            )
            
            if similarity > self.similarity_threshold:
                relevant_chunks.append(chunks[i])
                
        return relevant_chunks
    
    # nhận vào content đã được cào về từ một url cụ thể,
    # trích xuất ra được nội dung website và hàm 
    async def process_web_content(
        self,
        content: str,
        url: str,
        title: str,
        query: str,
        persist_dir: str = "./db",
    ) -> ProcessWebContentsResponse:
        """
        Process scraped content:
        1. Filter relevant content using LLM
        2. Chunk filtered content
        3. Store in Chroma
        """
        
        system_prompt = """
        You are a precise content extractor. Your job is to:
        1. NEVER generate new content
        2. ONLY extract and organize existing content
        3. Maintain the original structure (sections, lists)
        4. Keep important quotes with attribution
        """
        
        # chia nhỏ nội dung đã lọc thành các chunk vì câu trả lời có thể rất dài và LLM không xử lý hết context được vì máy mình yếu
        chunks = self.text_splitter.split_text(content)
        
        # filter chunks bằng similarity với query
        relevant_chunks = await self.filter_chunks_by_similarity(chunks, query)
        
        # Dùng LLM để filter từng chunk còn lại
        filtered_chunks: List[str] = []
        for chunk in relevant_chunks:
            filter_prompt = f"""
            {system_prompt}
            Content chunk: {chunk}
            Query: {query}
            Extract only information relevant to the query.
            """
            filtered = self.llm.invoke(filter_prompt)
            if filtered.strip():
                filtered_chunks.append(filtered)
        
        # Tạo ID riêng biệt cho mỗi chunk
        ids = [f"{url}-{i}" for i in range(len(filtered_chunks))]
        # Lưu vào vector database
        vectordb: Chroma = Chroma.from_texts(
            texts=filtered_chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name="articles",
            ids=ids,
            metadatas=[
                {
                    "url": url,
                    "query": query,
                    "chunk_id": i,
                    "total_chunks": len(filtered_chunks),
                    "original_chunk": chunks[i] # Lưu luôn text gốc
                } for i in range(len(filtered_chunks))
            ] # metadata cho mỗi chunk
        )
        
        return ProcessWebContentsResponse(
            title=title,
            filter_prompt=filter_prompt,
            filtered_content="\n\n".join(filtered_chunks),
            num_chunks=len(filtered_chunks),
            url=url,
            persist_dir=self.persist_dir
        )
    
    async def process_multiple_contents(
        self,
        contents: List[WebContentResponse],
        query: str
    ) -> Dict:
        """ Xử lý nhiều chunks đề phòng LLM không xử lý hết context vì giới hạn context """
        processed_contents = []
        
        for content in contents:
            processed = await self.process_web_content(
                content=content.content,
                url=content.metadata["url"] if content.metadata else "",
                title=content.title if content.title else "",
                query=query
            )
            
            processed_contents.append(processed)
            
        # lấy content từ vector DB, tìm kiếm những đoạn văn bản liên quan đến query
        combined_db: Chroma = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="articles"
        )
        
        # Tìm 5 đoạn văn bản liên quan nhất đến query
        results: List[Tuple[Document, float]] = combined_db.similarity_search_with_score(
            query=query,
            k=5 # Trả về 5 kết quả tốt nhất
        )
        
        # trích xuất nội dung từ kết quả tìm kiếm
        relevant_contents: List[str] = [doc.page_content for doc, _ in results]
        
        return {
            "processed_articles": processed_contents,
            "relevant_chunks": relevant_contents,
            "persist_dir": self.persist_dir
        }