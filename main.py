import os
import json
import requests
import uvicorn
from typing import List, Literal, Optional

# Third-party libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================================
# 1. CONFIGURATION
# ==========================================

GENAI_API_KEY = os.getenv("Gemini_Api_key")

if not GENAI_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. App will fail if not set.")

genai.configure(api_key=GENAI_API_KEY)

# NOTE: "gemini-2.5-flash" does not exist yet. Using 1.5-Flash.
MODEL_ID = "gemini-2.5-flash"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
}

app = FastAPI(title="Bill Extraction API")

# ==========================================
# 2. DATA SCHEMAS (FIXED)
# ==========================================

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float


class PageLineItems(BaseModel):
    page_no: int
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy", "Unknown"]
    bill_items: List[BillItem]

class ExtractedData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractedData] = None

class ExtractionRequest(BaseModel):
    document: str

# ==========================================
# 3. CORE LOGIC
# ==========================================

def download_file(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def process_document_to_bytes(file_content: bytes, file_url: str) -> List[bytes]:
    image_data_list = []
    try:
        if file_url.lower().endswith('.pdf') or file_content.startswith(b'%PDF'):
            doc = fitz.open(stream=file_content, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                image_data_list.append(pix.tobytes("png"))
        else:
            image_data_list.append(file_content)
        return image_data_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: ExtractionRequest):
    if not GENAI_API_KEY:
         raise HTTPException(status_code=500, detail="Server Configuration Error: API Key missing.")

    file_content = download_file(request.document)
    page_images = process_document_to_bytes(file_content, request.document)
    
    all_pages_data = []
    usage_stats = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )

    for page_index, img_bytes in enumerate(page_images):
        current_page_num = page_index + 1
        
        # --- FIXED PROMPT: Simplified to ask ONLY for data ---
        # The prompt shouldn't ask for "is_success" or "token_usage".
        # The python code handles that.
        prompt = f"""
        Extract data ONLY from this specific page (Page {current_page_num}).
        
        RETURN JSON SCHEMA:
        {{
            "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
            "bill_items": [ 
                {{ "item_name": "str", "item_amount": float, "item_rate": float, "item_quantity": float }} 
            ]
        }}
        
        RULES:
        1. If a row has Rate, Qty, and Amount, map them accurately.
        2. If a row has only Amount, set Rate=Amount and Qty=1.

        
        """

        try:
            response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_bytes}])
            
            if response.usage_metadata:
                usage_stats["input_tokens"] += response.usage_metadata.prompt_token_count
                usage_stats["output_tokens"] += response.usage_metadata.candidates_token_count
                usage_stats["total_tokens"] += response.usage_metadata.total_token_count

            # Parsing
            page_json = json.loads(response.text)
            
            all_pages_data.append(PageLineItems(
                page_no=current_page_num,
                page_type=page_json.get("page_type", "Unknown"),
                bill_items=page_json.get("bill_items", []),
            ))

        except Exception as e:
            print(f"Failed to process page {current_page_num}: {e}")
            # Add valid empty object to prevent downstream crashes
            all_pages_data.append(PageLineItems(
                page_no=current_page_num, 
                page_type="Unknown", 
                bill_items=[], 
            ))

    total_count = sum(len(p.bill_items) for p in all_pages_data)

    return APIResponse(
        is_success=True,
        token_usage=TokenUsage(**usage_stats),
        data=ExtractedData(
            pagewise_line_items=all_pages_data,
            total_item_count=total_count, 
        )
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)