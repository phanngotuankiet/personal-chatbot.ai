from tools.translation_tool import TranslationTool
from multiprocessing import freeze_support
from types_api.types_api import TranslateRequest

async def run_task_translate(req: TranslateRequest):
    freeze_support()  # Cần thiết cho multiprocessing
    
    # 1. Khởi tạo TranslationTool
    translator_tool = TranslationTool()

    # 2. Sử dụng tool trực tiếp
    try:
        result = translator_tool(
            text=req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang
        )
        
        print(f"Kết quả dịch: {result}")
        
        return {
            "result": result
        }
    except Exception as e:
        print(f"Lỗi khi dịch: {str(e)}")
        raise e
    
# export
__all__: list[str] = ["run_task_translate"]    