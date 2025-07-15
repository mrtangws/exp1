#!/usr/bin/env python3
"""
Predict invoice items for cancer treatments using OpenAI
"""

import json
import csv
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_historical_data(json_file):
    """Load historical cancer treatments and invoice items"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_inventory(csv_file):
    """Load available inventory items"""
    items = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assuming first column is item name
            item_name = list(row.values())[0] if row else ""
            if item_name:
                items.append(item_name.strip())
    return items

def predict_invoice_items(treatments, historical_file="invoice-sample.json", inventory_file="stock.csv"):
    """Predict invoice items for given cancer treatments"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load context data
    historical_data = load_historical_data(historical_file)
    inventory_items = load_inventory(inventory_file)
    
    # Limit context to avoid token limits
    historical_context = json.dumps(historical_data[:3], indent=2)  # Only first 3 examples
    inventory_context = "\n".join(inventory_items[:50])  # Only first 50 items
    
    results = []
    
    for treatment in treatments:
        prompt = f"""You are an expert Oncologist, who treat cancer patients. Given a cancer treatment, predict the invoice items that is required.

Historical cancer treatments and invoice items:
{historical_context}

Available inventory items in clinic:
{inventory_context}

Cancer treatment: {treatment}

Predict the invoice items required for this treatment. Return ONLY a JSON object in this exact format:
{{
  "cancer_treatment": "{treatment}",
  "invoice_items": [
    {{"id": "ITEM_NAME", "quantity": 1}},
    {{"id": "ITEM_NAME", "quantity": 2}}
  ]
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # Get response content
            content = response.choices[0].message.content.strip()
            # print(f"Raw response for {treatment}: {content[:200]}...")  # Debug
            
            # Try to extract JSON from response
            try:
                prediction = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    prediction = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            results.append(prediction)
            
        except Exception as e:
            print(f"Error predicting for {treatment}: {e}")
            results.append({
                "cancer_treatment": treatment,
                "invoice_items": [],
                "error": str(e)
            })
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_invoice.py 'TREATMENT1' 'TREATMENT2' ...")
        print("Example: python predict_invoice.py 'CA BREAST -- C1 HERCEPTIN' 'CA COLON -- XELOX'")
        sys.exit(1)
    
    treatments = sys.argv[1:]
    predictions = predict_invoice_items(treatments)
    
    print(json.dumps(predictions, indent=2))