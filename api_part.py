import requests
import re
import pandas as pd

img = 'IMAGE.jpg'

api = 'https://api.api-ninjas.com/v1/imagetotext'


api_url = 'https://api.api-ninjas.com/v1/imagetotext'
image_file_descriptor = open(img, 'rb')
files = {'image': image_file_descriptor}
r = requests.post(api_url, files=files, headers={'X-Api-Key': 'YOUR_API_KEY'})
#print(r.json())


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
    lines = text.split('\n')

    item_price_pattern = re.compile(r'(.+?)\s*[-]?\s*\$?(\d+\.\d{2})')  # Pattern to extract item-price pairs
    items_prices = {}

    # Define reject keywords regex for filtering out unwanted content
    reject_keywords = re.compile(
        r'\b(Field|HWY|Street|Road|Ave|Blvd)\b|\d{1,2}/\d{1,2}/\d{4}|\(\d{3}\)\s?\d{3}-\d{4}|[A-Za-z]+\.[A-Za-z]+',
        re.IGNORECASE
    )

    common_labels_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(label) for label in ["Table", "Dine In", "Take", "Order", "Check"]) + r')\b', re.IGNORECASE)

    for line in lines:
        print(f"Processing line: {line}")

        if common_labels_pattern.search(line):
            parts = common_labels_pattern.split(line, maxsplit=1)[-1]  
        else:
            parts = line  # If no common label, process the whole line as one part

        # Split the remaining part into individual components (words or phrases)
        parts = parts.split()

        filtered_parts = []
        for part in parts:
            if not re.search(reject_keywords, part):
                filtered_parts.append(part)

        print(f"Filtered parts: {filtered_parts}")
        
        filtered_line = ' '.join(filtered_parts)
        print(f"Filtered line: {filtered_line}")
        
        matches = item_price_pattern.findall(filtered_line)
        for item, price in matches:
            item = item.strip()
            if item: 
                items_prices[item] = float(price)

    return items_prices



def comparison(response_data):
    extracted_text = "\n".join([item["text"] for item in response_data]) 
    matches = re.findall(r'(.+)\s*[-]?\s*\$?(\d+\.\d{2})', extracted_text)
    compared_receipt_data = {item.strip(): float(price) for item, price in matches}

    return(compared_receipt_data)

response_data = r.json()


receipt_data = extract_prices(response_data)

print('receipt data:')
print(receipt_data)

df = pd.DataFrame(receipt_data.items(), columns=['Item', 'Price'])
compared_receipt = comparison(response_data)

print(df)
print(compared_receipt)
    
