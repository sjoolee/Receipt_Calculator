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

import re

def extract_prices(ocr_response):
    text = correct_ocr_errors(ocr_response)  # Correct OCR errors if necessary
    lines = text.split('\n')

    item_price_pattern = re.compile(r'(.+?)\s*[-]?\s*\$?(\d+\.\d{2})')  # Pattern to extract item-price pairs
    items_prices = {}

    # Define reject keywords regex for filtering out unwanted content
    reject_keywords = re.compile(
        r'\b(Field|HWY|Street|Road|Ave|Blvd)\b|\d{1,2}/\d{1,2}/\d{4}|\(\d{3}\)\s?\d{3}-\d{4}|[A-Za-z]+\.[A-Za-z]+',
        re.IGNORECASE
    )

    # Create a regex pattern to match any of the common labels
    common_labels_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(label) for label in ["Table", "Dine In", "Order", "Check"]) + r')\b', re.IGNORECASE)

    for line in lines:
        print(f"Processing line: {line}")

        # Split the line at the common labels if they are present, and keep the part after the label
        if common_labels_pattern.search(line):  # Check if any common label is in the line
            # Find the part after the first common label
            parts = common_labels_pattern.split(line, maxsplit=1)[-1]  # Get the part after the first match
        else:
            parts = line  # If no common label, process the whole line as one part

        # Split the remaining part into individual components (words or phrases)
        parts = parts.split()

        # Filter out parts that match reject keywords
        filtered_parts = []
        for part in parts:
            # Check if the part matches any of the reject keywords
            if not re.search(reject_keywords, part):
                filtered_parts.append(part)

        print(f"Filtered parts: {filtered_parts}")  # Debugging the filtered parts

        # Rebuild the filtered line
        filtered_line = ' '.join(filtered_parts)
        print(f"Filtered line: {filtered_line}")  # Debugging the filtered line

        # Extract valid item-price pairs from the filtered line
        matches = item_price_pattern.findall(filtered_line)
        for item, price in matches:
            item = item.strip()
            if item:  # Only add valid items
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
    
