import os
import json
import time
import requests
import uvicorn # Added for local testing
from typing import List, Literal, Optional

# Third-party libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================================
# 1. CONFIGURATION
# ==========================================

# [CRITICAL CHANGE] Never hardcode API keys in code committed to Git!
# On Render, you will set this in the "Environment" tab.
GENAI_API_KEY = os.getenv("Gemini_Api_key") 

if not GENAI_API_KEY:
    # Fallback for local testing only - prevents crash on deploy if key is missing
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GENAI_API_KEY)

MODEL_ID = "gemini-2.5-flash"

# Safety: Block nothing (Medical/Financial docs often trigger false positives)
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

# Note: On Render, this directory is ephemeral (files delete on restart)
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

app = FastAPI(title="Bill Extraction API")

# ==========================================
# 2. DATA SCHEMAS
# ==========================================

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class SubTotalItem(BaseModel):
    section_name: str
    amount: float

class PageLineItems(BaseModel):
    page_no: int
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy", "Unknown"]
    bill_items: List[BillItem]
    sub_totals: List[SubTotalItem]

class ExtractedData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_line_items_found: int
    grand_total_amount: float

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
        
        prompt = f"""
        Extract data ONLY from this specific page (Page {current_page_num}).
        RETURN JSON SCHAME:
        {{
            "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
            "bill_items": [ {{ "item_name": "str", "item_amount": float, "item_rate": float, "item_quantity": float }} ],
        }}
        RULES:
        1. If a row has Rate, Qty, and Amount, map them accurately.
        2. If a row has only Amount, set Rate=Amount and Qty=1.

        Expample JSON Output:
        {
        "is_success": "boolean", // If Status code 200 and following valid schema, then true
        "token_usage": {
            "total_tokens": "integer", // Cumulative Tokens from all LLM calls
            "input_tokens": "integer", // Cumulative Tokens from all LLM calls
            "output_tokens": "integer" // Cumulative Tokens from all LLM calls
        },
        "data": {
            "pagewise_line_items": [
            {
                "page_no": "string",
                "page_type": "Bill Detail | Final Bill | Pharmacy",
                "bill_items": [
                {
                    "item_name": "string", // Exactly as mentioned in the bill
                    "item_amount": "float", // Net Amount of the item post discounts as mentioned in the bill
                    "item_rate": "float", // Exactly as mentioned in the bill
                    "item_quantity": "float" // Exactly as mentioned in the bill
                }
                ]
            }
            ],
            "total_item_count": "integer" // Count of items across all pages
        }
        }
        
        """

        try:
            response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_bytes}])
            
            if response.usage_metadata:
                usage_stats["input_tokens"] += response.usage_metadata.prompt_token_count
                usage_stats["output_tokens"] += response.usage_metadata.candidates_token_count
                usage_stats["total_tokens"] += response.usage_metadata.total_token_count

            page_json = json.loads(response.text)
            
            all_pages_data.append(PageLineItems(
                page_no=current_page_num,
                page_type=page_json.get("page_type", "Unknown"),
                bill_items=page_json.get("bill_items", []),
                # sub_totals=page_json.get("sub_totals", [])
            ))

        except Exception as e:
            print(f"Failed to process page {current_page_num}: {e}")
            all_pages_data.append(PageLineItems(page_no=current_page_num, page_type="Unknown", bill_items=[], sub_totals=[]))

    # grand_total = sum(item.item_amount for p in all_pages_data for item in p.bill_items)
    total_count = sum(len(p.bill_items) for p in all_pages_data)

    return APIResponse(
        is_success=True,
        token_usage=TokenUsage(**usage_stats),
        data=ExtractedData(
            pagewise_line_items=all_pages_data,
            total_item_count=total_count,
            # grand_total_amount=round(grand_total, 2)
        )
    )

# [CRITICAL] This allows Render to start the app correctly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)