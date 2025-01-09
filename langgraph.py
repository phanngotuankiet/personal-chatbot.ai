from typing import TypedDict, List, Dict
from langchain_core.documents.base import Document
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langgraph.graph.state import CompiledStateGraph

# Định nghĩa kiểu dữ liệu cho state
class ResearchState(TypedDict):
    query: str                    # Câu hỏi nghiên cứu
    documents: List[str]          # Tài liệu thu thập được
    relevant_chunks: List[str]    # Đoạn văn bản liên quan
    analysis: Dict[str, str]      # Phân tích từng phần
    final_summary: str            # Tổng hợp cuối cùng
    current_step: str             # Bước hiện tại
    errors: List[str]             # Lưu lỗi nếu có

# Các hàm xử lý cho từng node
def collect_documents(state: ResearchState) -> ResearchState:
    """Thu thập tài liệu từ nhiều nguồn dựa trên query"""
    # Giả lập việc tìm kiếm tài liệu
    state["documents"] = [
        "Tài liệu từ nguồn 1...",
        "Tài liệu từ nguồn 2...",
    ]
    return state

def process_documents(state: ResearchState) -> ResearchState:
    """Xử lý và chia nhỏ tài liệu"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = []
    for doc in state["documents"]:
        chunks.extend(text_splitter.split_text(doc))
    
    # Tạo vector store
    vectorstore: Chroma = Chroma.from_texts(
        chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    
    # Tìm các đoạn liên quan nhất
    relevant_docs: List[Document] = vectorstore.similarity_search(state["query"], k=5)
    state["relevant_chunks"] = [doc.page_content for doc in relevant_docs]
    return state

def analyze_information(state: ResearchState) -> ResearchState:
    """Phân tích thông tin từ các đoạn văn bản"""
    llm = OllamaLLM(model="qwen2:latest")
    
    analyses = {}
    for chunk in state["relevant_chunks"]:
        analysis: str = llm.invoke(
            f"Phân tích đoạn văn bản sau liên quan đến câu hỏi: {state['query']}\n\n{chunk}"
        )
        analyses[chunk[:100]] = analysis
    
    state["analysis"] = analyses
    return state

def generate_summary(state: ResearchState) -> ResearchState:
    """Tạo bản tổng hợp cuối cùng"""
    llm = OllamaLLM(model="qwen2:latest")
    
    all_analyses: str = "\n\n".join(state["analysis"].values())
    summary_prompt: str = f"""
    Tổng hợp các phân tích sau để trả lời câu hỏi: {state['query']}
    
    Các phân tích:
    {all_analyses}
    
    Hãy tạo một bản tổng hợp ngắn gọn, súc tích và có cấu trúc rõ ràng.
    """
    
    state["final_summary"] = llm.invoke(summary_prompt)
    return state

# Tạo workflow
def create_research_workflow() -> CompiledStateGraph:
    workflow = StateGraph(ResearchState)
    
    # Thêm các node
    workflow.add_node("collect_documents", collect_documents)
    workflow.add_node("process_documents", process_documents)
    workflow.add_node("analyze_information", analyze_information)
    workflow.add_node("generate_summary", generate_summary)
    
    # Định nghĩa luồng
    workflow.set_entry_point("collect_documents")
    workflow.add_edge("collect_documents", "process_documents")
    workflow.add_edge("process_documents", "analyze_information")
    workflow.add_edge("analyze_information", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    # Compile workflow
    return workflow.compile()

# Sử dụng workflow
def main():
    workflow: CompiledStateGraph = create_research_workflow()
    
    # Khởi tạo state
    initial_state = ResearchState(
        query="Tác động của AI đến thị trường việc làm trong 5 năm tới?",
        documents=[],
        relevant_chunks=[],
        analysis={},
        final_summary="",
        current_step="",
        errors=[]
    )
    
    # Chạy workflow
    final_state = workflow.invoke(initial_state)
    
    # In kết quả
    print("Tổng hợp cuối cùng:")
    print(final_state["final_summary"])

if __name__ == "__main__":
    main()