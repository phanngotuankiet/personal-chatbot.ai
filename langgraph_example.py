from typing import Any, TypedDict, List, Dict, Optional, Union, Literal
from langchain_core.documents.base import Document
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langgraph.graph.state import CompiledStateGraph
from enum import Enum
from dataclasses import dataclass, field

# Định nghĩa các loại yêu cầu có thể có
class QueryType(Enum):
    DATABASE_SEARCH = "database_search"
    GENERAL_QUESTION = "general_question"
    UNKNOWN = "unknown"

@dataclass
class UserQuery:
    raw_text: str
    query_type: QueryType
    extracted_keywords: List[str] = field(default_factory=list)
    
def classify_user_query(query: str) -> UserQuery:
    """
    Sử dụng LLM để phân loại yêu cầu của user một cách thông minh hơn
    """
    llm = OllamaLLM(model="deepseek-r1:1.5b-qwen-distill-q8_0")
    
    classification_prompt: str = f"""
    Hãy phân loại câu hỏi sau vào một trong hai loại:
    1. DATABASE_SEARCH: Nếu người dùng có ý định tìm kiếm thông tin từ database/báo cáo/dữ liệu có sẵn
    2. GENERAL_QUESTION: Nếu là câu hỏi tổng quát cần trả lời dựa trên kiến thức chung

    Câu hỏi: "{query}"

    Phân tích:
    - Nếu câu hỏi yêu cầu tra cứu dữ liệu cụ thể, số liệu, báo cáo -> DATABASE_SEARCH
    - Nếu câu hỏi mang tính tổng quát, yêu cầu phân tích, dự đoán -> GENERAL_QUESTION

    Trả lời CHÍNH XÁC một trong hai từ sau (không thêm bất kỳ ký tự nào khác):
    DATABASE_SEARCH
    GENERAL_QUESTION
    """

    # Lấy kết quả phân loại từ LLM và xử lý để lấy chỉ kết quả cuối cùng
    raw_response: str = llm.invoke(classification_prompt).strip()
    
    # Xử lý response để lấy kết quả cuối cùng
    if "<think>" in raw_response:
        # Tìm dòng cuối cùng sau tag </think>
        classification: str = raw_response.split("</think>")[-1].strip()
    else:
        classification: str = raw_response
        
    # Đảm bảo chỉ lấy một trong hai giá trị hợp lệ
    if "DATABASE_SEARCH" in classification:
        classification = "DATABASE_SEARCH"
    elif "GENERAL_QUESTION" in classification:
        classification = "GENERAL_QUESTION"
    else:
        # Fallback nếu không match
        classification = "GENERAL_QUESTION"
    
    print(f"\nRaw response from LLM: {raw_response}")
    print(f"Cleaned classification: {classification}\n")

    # Phân tích từ khóa nếu là DATABASE_SEARCH
    if classification == "DATABASE_SEARCH":
        print("\n\n Check coi từ khoá có phải là DATABASE_SEARCH: ",classification,"\n\n")
        keywords_prompt: str = f"""
        Từ câu hỏi sau, hãy trích xuất các từ khóa quan trọng để tìm kiếm trong database.
        Chỉ liệt kê các từ khóa, cách nhau bằng dấu phẩy.
        
        Câu hỏi: "{query}"
        
        Các từ khóa:
        """
        
        # Lấy từ khóa từ LLM và chuyển thành list
        keywords_str: str = llm.invoke(keywords_prompt).strip()
        extracted_keywords: List[str] = [k.strip() for k in keywords_str.split(',')]
        
        return UserQuery(
            raw_text=query,
            query_type=QueryType.DATABASE_SEARCH,
            extracted_keywords=extracted_keywords
        )
    else:
        return UserQuery(
            raw_text=query,
            query_type=QueryType.GENERAL_QUESTION
        )

# Định nghĩa kiểu dữ liệu cho state
class ResearchState(TypedDict):
    query: str                     # Câu hỏi nghiên cứu
    query_type: QueryType         # Thêm trường này
    documents: List[str]          
    relevant_chunks: List[str]    
    analysis: Dict[str, str]      
    final_summary: str            
    current_step: str             
    errors: List[str]             

# Thêm vào đầu file, sau phần import
def mock_database_query(query: str) -> Any:
    """
    Giả lập database chứa các bài báo và báo cáo về AI
    """
    mock_documents = {
        "ai_impact": [
            """
            Theo báo cáo của World Economic Forum 2024, AI sẽ tự động hóa khoảng 85 triệu việc làm 
            và tạo ra 97 triệu việc làm mới vào năm 2025. Các ngành chịu tác động mạnh nhất bao gồm:
            kế toán, dịch vụ khách hàng, và công việc hành chính. Tuy nhiên, nhu cầu về các vị trí
            như kỹ sư AI, chuyên gia phân tích dữ liệu, và quản lý quy trình tự động hóa sẽ tăng mạnh.
            """,
            
            """
            Khảo sát từ McKinsey 2023 cho thấy 56% công ty đang thử nghiệm hoặc triển khai AI 
            trong hoạt động kinh doanh. Điều này dẫn đến nhu cầu đào tạo lại kỹ năng cho 40% 
            lực lượng lao động hiện tại. Các kỹ năng được ưu tiên bao gồm: tư duy phản biện,
            khả năng làm việc với AI, và kỹ năng giải quyết vấn đề phức tạp.
            """,
            
            """
            Theo dự báo của LinkedIn, đến năm 2028, 75% công việc sẽ yêu cầu ít nhất kiến thức
            cơ bản về AI. Các ngành như y tế, giáo dục, và luật sẽ tích hợp AI như một công cụ
            hỗ trợ, không phải thay thế hoàn toàn con người. Xu hướng này tạo ra mô hình làm việc
            "AI-human collaboration" thay vì "AI replacement".
            """
        ],
        
        "job_market": [
            """
            Báo cáo từ Bureau of Labor Statistics dự báo tăng trưởng 13% cho các vị trí liên quan
            đến AI và máy học trong giai đoạn 2023-2028. Mức lương trung bình cho các vị trí này
            cao hơn 45% so với mức lương trung bình của ngành công nghệ thông tin.
            """,
            
            """
            Các startup AI đã tạo ra hơn 250,000 việc làm mới trong năm 2023, tập trung vào các
            lĩnh vực: phát triển mô hình AI, xử lý dữ liệu, và tích hợp AI vào các giải pháp
            doanh nghiệp. Dự kiến con số này sẽ tăng gấp đôi vào năm 2025.
            """
        ],
        
        "skills_development": [
            """
            Google và Microsoft đã công bố các chương trình đào tạo AI miễn phí, dự kiến đào tạo
            2 triệu người trong 3 năm tới. Các kỹ năng được đào tạo bao gồm: lập trình Python,
            xử lý ngôn ngữ tự nhiên, và đạo đức AI.
            """
        ]
    }
    
    # Phân loại query và lấy keywords
    user_query: UserQuery = classify_user_query(query)
    
    # Nếu không phải DATABASE_SEARCH, return empty list
    if user_query.query_type != QueryType.DATABASE_SEARCH:
        return []
    
    relevant_docs: List[str] = []
    
    # Sử dụng extracted_keywords thay vì split query
    for category, documents in mock_documents.items():
        for doc in documents:
            # Kiểm tra nếu document chứa bất kỳ keyword nào
            if any(keyword.lower() in doc.lower() for keyword in user_query.extracted_keywords):
                relevant_docs.append(doc)
    
    return relevant_docs

def collect_documents(state: ResearchState) -> ResearchState:
    """Thu thập tài liệu dựa vào loại query"""
    
    # Phân tích yêu cầu của user
    user_query: UserQuery = classify_user_query(state["query"])
    state["query_type"] = user_query.query_type
    
    if user_query.query_type == QueryType.DATABASE_SEARCH:
        # Nếu user yêu cầu tìm trong database
        documents = mock_database_query(state["query"])
        print("Documents giả lập tìm được: ", documents)
        if documents:
            state["documents"] = documents
        else:
            state["errors"].append(f"Không tìm thấy tài liệu phù hợp trong database cho query: {state['query']}")
            state["documents"] = ["Không có dữ liệu phù hợp trong database."]
    else:
        # Nếu là câu hỏi thông thường, không cần query database
        state["documents"] = [
            "Đây là câu trả lời chung, không cần tra cứu database...",
            "LLM sẽ trả lời dựa trên kiến thức có sẵn..."
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
    llm = OllamaLLM(model="deepseek-r1:1.5b-qwen-distill-q8_0")
    
    analyses = {}
    for chunk in state["relevant_chunks"]:
        # Điều chỉnh prompt dựa vào loại query
        if state["query_type"] == QueryType.DATABASE_SEARCH:
            prompt = f"""
            Phân tích đoạn trích từ database sau đây liên quan đến câu hỏi: {state['query']}
            
            Đoạn trích: {chunk}
            
            Hãy tập trung vào các số liệu và dữ liệu cụ thể từ các báo cáo.
            """
        else:
            prompt: str = f"""
            Phân tích vấn đề sau: {state['query']}
            
            Dựa trên thông tin: {chunk}
            
            Hãy đưa ra nhận định tổng quan.
            """
        
        analysis = llm.invoke(prompt)
        analyses[chunk[:100]] = analysis
    
    state["analysis"] = analyses
    return state

def generate_summary(state: ResearchState) -> ResearchState:
    """Tạo bản tổng hợp cuối cùng"""
    llm = OllamaLLM(model="deepseek-r1:1.5b-qwen-distill-q8_0")
    
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
def main() -> None:
    workflow: CompiledStateGraph = create_research_workflow()
    
    # queries: List[str] = [
    #     "Tra trong database xem có báo cáo nào về tác động của AI đến việc làm không?",
    #     "AI sẽ ảnh hưởng thế nào đến tương lai việc làm?",
    #     "Tìm các báo cáo về xu hướng thị trường lao động AI",
    # ]

    print("\n=== AI Research Assistant ===")
    print("Bạn có thể:")
    print("1. Tra cứu trong database bằng cách dùng từ khóa như 'tra cứu', 'tìm báo cáo', 'tra trong database'")
    print("2. Hỏi câu hỏi thông thường về AI")
    print("3. Gõ 'exit' hoặc 'quit' để thoát\n")
    
    while True:
        try:
            # Nhận input từ người dùng
            query: str = input("\nNhập câu hỏi của bạn: ").strip()
            
            # Kiểm tra nếu user muốn thoát
            if query.lower() in ['exit', 'quit', 'thoát']:
                print("\nCảm ơn bạn đã sử dụng! Tạm biệt!")
                break
                
            if not query:  # Bỏ qua nếu input rỗng
                continue
                
            print("\nĐang xử lý câu hỏi của bạn...")
            
            # Khởi tạo state
            initial_state = ResearchState(
                query=query,
                query_type=QueryType.UNKNOWN,  # Sẽ được xác định trong collect_documents
                documents=[],
                relevant_chunks=[],
                analysis={},
                final_summary="",
                current_step="",
                errors=[]
            )
            
            # Chạy workflow
            final_state: Dict[str, Any] | Any = workflow.invoke(initial_state)
            
            # In kết quả
            print(f"\nLoại câu hỏi: {final_state['query_type'].value}")
            
            if final_state['errors']:  # Nếu có lỗi
                print("\nLỗi xảy ra:")
                for error in final_state['errors']:
                    print(f"- {error}")
                    
            print("\nKết quả:")
            print(final_state["final_summary"])
            
        except KeyboardInterrupt:
            print("\n\nĐã nhận lệnh thoát. Tạm biệt!")
            break
            
        except Exception as e:
            print(f"\nCó lỗi xảy ra: {str(e)}")
            print("Vui lòng thử lại!")
            continue

if __name__ == "__main__":
    main()