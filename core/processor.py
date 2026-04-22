"""
processor.py — OCR & LLM Extraction
Handles reading text from receipt images (EasyOCR) and parsing
the text into structured JSON using Ollama (granite3.2-vision).
"""

import easyocr
import json
import re
from datetime import datetime
from ollama import Client

# Initialize EasyOCR reader (downloads models on first run)
reader = easyocr.Reader(["en"], gpu=False)


def extract_text_from_image(image_path: str) -> str:
    """Run EasyOCR on an image and return the concatenated text."""
    try:
        results = reader.readtext(image_path, detail=0)
        if not results:
            return ""
        return " ".join(results)
    except Exception as e:
        print(f"[processor] OCR error: {e}")
        return ""


def sanitize_total(value) -> float:
    """Convert messy total strings into a clean float.
    Handles cases like '9,58' -> 9.58,  '1.234,56' -> 1234.56, etc.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    # Remove currency symbols and whitespace
    text = re.sub(r"[^\d,.\-]", "", text)

    if not text:
        return 0.0

    # If both comma and dot exist, figure out which is the decimal separator
    if "," in text and "." in text:
        # e.g. "1,234.56" or "1.234,56"
        if text.rfind(",") > text.rfind("."):
            # Comma is the decimal: "1.234,56" -> "1234.56"
            text = text.replace(".", "").replace(",", ".")
        else:
            # Dot is the decimal: "1,234.56" -> "1234.56"
            text = text.replace(",", "")
    elif "," in text:
        # Only commas — could be "1,234" (thousands) or "9,58" (decimal)
        parts = text.split(",")
        if len(parts[-1]) == 3 and len(parts) > 1:
            # Likely thousands separator: "1,234"
            text = text.replace(",", "")
        else:
            # Likely decimal: "9,58"
            text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return 0.0


def sanitize_date(date_str: str) -> str:
    """Ensure the date is realistic. Fix common OCR year errors where
    '1' is read as '4' or '8', e.g., '2048' or '2088' instead of '2018'.
    """
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        current_year = datetime.now().year
        
        # If year is in the future, it's likely an OCR error
        if dt.year > current_year:
            corrected_year = dt.year
            # Common OCR typos: 4 instead of 1, 8 instead of 1, etc.
            # e.g., 2048 -> 2018. We just subtract 30 or 70 if it ends in 8.
            if dt.year >= 2040 and dt.year <= 2099:
                # If the last digit is reasonable, maybe it was a 10s typo
                str_year = str(dt.year)
                # Just replace the 3rd digit (e.g. 4 -> 1 or 2)
                # Let's just default to current_year or a safe guess if it's way off.
                # Actually, a simple heuristic: 2048 -> 2018
                if str_year[2] in ['4', '8', '9']:
                    str_year = str_year[:2] + '1' + str_year[3:]
                    corrected_year = int(str_year)
            
            # If it's still in the future after heuristic, clamp to current year
            if corrected_year > current_year:
                corrected_year = current_year
                
            return f"{corrected_year:04d}-{dt.month:02d}-{dt.day:02d}"
        
        return date_str
    except ValueError:
        # If it doesn't match YYYY-MM-DD, just return what the LLM gave us
        return date_str


def parse_with_ollama(raw_text: str) -> dict | None:
    """Send raw OCR text to Ollama and get structured JSON back.
    Uses granite3.2-vision for extraction.
    Returns a dict with keys: merchant, date, total, currency, category
    or None on failure.
    """
    if not raw_text.strip():
        return None

    client = Client(host="http://localhost:11434")

    prompt = f"""
    You are a strict and precise data extractor. Convert this raw OCR text from a receipt into a valid JSON object.
    TEXT: {raw_text}
    
    CRITICAL INSTRUCTION: 
    - Make sure to read the whole receipt and understand the context.
    - NEVER calculate numbers yourself (like adding taxes or tips). ONLY extract the exact numbers printed in the text.
    - Find the TRUE FINAL TOTAL. 
    - WARNING: Receipts often have lines like "Tax Total SALES TAX ... 453.01". DO NOT use these. Ignore any number associated with "TAX", "Tax Total", or "SALES TAX".
    - WARNING: Do NOT extract "CASH", "TENDERED", or "CHANGE" amounts as the total. (e.g., if you see "TOTAL 175,000 CASH 200,000", the total is 175,000).
    - The actual final total is usually labeled simply "Total" and appears right before the Date or near the very bottom (e.g., "Total 184 .47" -> 184.47).
    - If you see a comma used as a decimal (e.g., 9,58), convert it to a period (9.58).
    - Ensure the 'total' is a valid FLOAT number.
    - Ensure the 'date' is a realistic past or present date. Fix obvious OCR typos (e.g., '2048' should probably be '2018').
    - If no currency symbol is explicitly printed, try to infer it from the location or merchant name (e.g., IDR for Indonesia, USD for US). If completely unsure, leave it as an empty string "".

    REQUIRED FIELDS:
    - "merchant": Name of the store
    - "date": Date in YYYY-MM-DD format
    - "total": The TRUE final total amount (as a decimal number) e.g., 0.00 .
    - "currency": The currency symbol or code (e.g., "$", "INR", "€", "₹")
    - "category": One of [Food, Transport, Shopping, Utilities, Other]
    

    Return ONLY the JSON. No chat.
    """

    try:
        response = client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )

        content = response["message"]["content"]

        # Extract JSON from the response (model may include extra text)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            # Sanitize values through our robust parsers
            data["total"] = sanitize_total(data.get("total", 0))
            data["date"] = sanitize_date(data.get("date", ""))
            return data
        return None
    except Exception as e:
        print(f"[processor] Ollama error: {e}")
        return None