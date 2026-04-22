import easyocr
import ollama
import json
import re
from ollama import Client

# Initialize EasyOCR (Downloads models on first run)
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    # Perform OCR
    results = reader.readtext(image_path, detail=0)
    return " ".join(results)

def parse_with_ollama(raw_text):
    client = Client(host='http://localhost:11434')
    prompt = f"""
    You are a professional accountant. Convert this raw OCR text from a receipt into a valid JSON object.
    TEXT: {raw_text}
    
    CRITICAL INSTRUCTION: 
    - Look closely at the 'total' amount. 
    - If you see a comma used as a decimal (e.g., 9,58), convert it to a period (9.58).
    - Ensure the 'total' is a valid FLOAT number.

    REQUIRED FIELDS:
    - "merchant": Name of the store
    - "date": Date in YYYY-MM-DD format
    - "total": The final total amount (as a decimal number) e.g., 0.00 .
    - "currency": The currency symbol or code (e.g., "$", "INR", "€", "₹")
    - "category": One of [Food, Transport, Shopping, Utilities, Other]
    

    Return ONLY the JSON. No chat.
    """
    
    # We use a text model like granite3-dense because it's much faster for text-only tasks
    response = client.chat(
        model='granite3.2-vision:latest ', 
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    # Clean the response to ensure it's valid JSON
    content = response['message']['content']
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return None