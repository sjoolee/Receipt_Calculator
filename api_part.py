import requests
import re
import pandas as pd

img = 'IMAGE.jpg'

api = 'https://api.api-ninjas.com/v1/imagetotext'


api_url = 'https://api.api-ninjas.com/v1/imagetotext'
image_file_descriptor = open(img, 'rb')
files = {'image': image_file_descriptor}
r = requests.post(api_url, files=files, headers={'X-Api-Key': 'YOUR_API_KEY'})
print(r.json())


def correct_ocr_errors(text):
    """Fix common OCR errors in extracted text."""
    if isinstance(text, list):  
        text = " ".join([t['text'] for t in text])  

    corrections = {
        r'\bus\b': '4',      # Fix misread numbers (us5.90 → 45.90)
        r'[,;]': '.',        # Replace commas/semicolons with periods
        r'O(\d)': r'0\1',    # Fix zero misreads (O4.90 → 04.90)
        r'(\d+)\s+(\d{2})': r'\1.\2'  # Fix misplaced decimals (45 90 → 45.90)
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)

    return text

def extract_prices(ocr_response):
    text = correct_ocr_errors(ocr_response)

    text_lines = text.splitlines()
    
    # Regex to capture item names and prices (exclude location and irrelevant lines)
    items_prices = {}
    for line in text_lines:
        # Skip lines with location info or other irrelevant details
        if re.search(r'\bField\b.*\bHWY\b|\d{1,2}/\d{1,2}/\d{4}', line):
            continue
        
        # Match items and prices
        match = re.search(r'(\d+)\s*(\D+?)\s*(\d+\.\d{2})', line)

    matches = re.findall(r'(.+?)\s*[-]?\s*\$?(\d+\.\d{2})', text)

    items_prices = {}
    for item, price in matches:
        item = item.strip()
        items_prices[item] = float(price)

    return items_prices


def comparison(response_data):
    extracted_text = "\n".join([item["text"] for item in response_data]) 
    matches = re.findall(r'(.+)\s*[-]?\s*\$?(\d+\.\d{2})', extracted_text)
    compared_receipt_data = {item.strip(): float(price) for item, price in matches}

    return(compared_receipt_data)

response_data = r.json()

receipt_data = extract_prices(response_data)
df = pd.DataFrame(receipt_data.items(), columns=['Item', 'Price'])
compared_receipt = comparison(response_data)

print(df)
print(compared_receipt)
    
