from translation_tool import TranslationTool
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # Cần thiết cho multiprocessing
    
    # 1. Khởi tạo TranslationTool
    translator_tool = TranslationTool()

    # 2. Sử dụng tool trực tiếp
    try:
        result = translator_tool(
            text="Hello, how are you?",
            source_lang="en",
            target_lang="vi"
        )
        
        print(f"Kết quả dịch: {result}")
    except Exception as e:
        print(f"Lỗi: {str(e)}")