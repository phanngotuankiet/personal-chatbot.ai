from smolagents import CodeAgent, HfApiModel

# Load tool tự define vào
from translation_tool import TranslationTool

# Tải các biến từ environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Lấy token từ environment variables
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set in environment variables")

# Khai báo tool
translator = TranslationTool()

# Khai báo model từ huggingface và lấy token để dùng
model = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # Model chuyên biệt cho en-vi
    token=hf_token
)

# Khởi tạo agent
agent = CodeAgent(
    tools=[translator], 
    model=model,
    add_base_tools=False  # Không cần thêm base tools vì chỉ focus vào translation
)

# Test translation
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    try:
        result = agent.run("Translate 'Hello, how are you?' from English to Vietnamese")
        print(f"Translation result: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")

# export
__all__ = ["agent", "translator"]
