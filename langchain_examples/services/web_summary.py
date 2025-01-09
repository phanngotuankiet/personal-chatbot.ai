# hàm xử lý tóm tắt nội dung từ website thông qua url
from fastapi import HTTPException
from langchain_ollama import ChatOllama, OllamaLLM
from types_api.types_api import SummarizeRequest, SummarizeResponse
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

async def web_summary(req: SummarizeRequest) -> SummarizeResponse:
    try:
        loader = WebBaseLoader(req.url)
        docs = loader.load()
        
        if not docs:
            raise HTTPException(status_code=404, detail="No content found")
        
        # init LLM
        # llm = OllamaLLM(model=req.model or "llama2-uncensored")
        llm = ChatOllama(model=req.model or "llama2-uncensored")
        # chain từ langchain để summarize
        chain = load_summarize_chain(
            llm=llm, 
            chain_type=req.chain_type or "stuff", 
            verbose=True
        )
        
        # thực hiện tóm tắt với các bước:
        # 1. Lấy nội dung từ 'docs'
        # 2. Tạo prompt lên model LLM yêu cầu tóm tắt
        # 3. Lưu vào 'result'
        result = chain.invoke(docs)
        
        return SummarizeResponse(
            success=True,
            summary=str(result),
            metadata=req
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
__all__: list[str] = ["web_summary"]