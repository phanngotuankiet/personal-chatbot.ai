from smolagents import Tool
from transformers import MarianMTModel, MarianTokenizer

class TranslationTool(Tool):
    name = "translator"
    description = "Dịch văn bản giữa các ngôn ngữ. Hỗ trợ các ngôn ngữ: en, vi, fr, de, es, zh"
    inputs = {
        "text": {
            "type": "string", 
            "description": "Văn bản cần dịch"
        },
        "source_lang": {
            "type": "string",
            "description": "Mã ngôn ngữ nguồn (vd: en, vi, fr)" 
        },
        "target_lang": {
            "type": "string",
            "description": "Mã ngôn ngữ đích (vd: en, vi, fr)"
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        model_name = "Helsinki-NLP/opus-mt-en-vi"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def forward(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            print(f"Translating from {source_lang} to {target_lang}")
            print(f"Input text: {text}")
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            
            # Generate translation
            translated = self.model.generate(**inputs)
            
            # Decode
            result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            print(f"Translated text: {result}")
            return result

        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            print(error_msg) 
            raise Exception(error_msg)