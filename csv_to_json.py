#!/usr/bin/env python3
"""
Convert invoice CSV to JSON format
"""

import csv
import json

def csv_to_json(csv_file, output_file=None):
    """Convert invoice CSV to JSON format"""
    
    result = []
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if len(row) < 2:
                continue
                
            treatment = row[0].strip()
            items_string = row[1].strip()
            
            # Parse comma-separated items with quantities
            items = []
            parts = items_string.split(',')
            
            # Process pairs of (item, quantity)
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    item_id = parts[i].strip()
                    try:
                        quantity = int(parts[i + 1].strip())
                    except:
                        quantity = 1
                    
                    if item_id:
                        items.append({
                            "id": item_id,
                            "quantity": quantity
                        })
            
            if treatment and items:
                result.append({
                    "cancer_treatment": treatment,
                    "invoice_items": items
                })
    
    # Output
    json_output = json.dumps(result, indent=2)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"JSON saved to {output_file}")
    else:
        print(json_output)
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python csv_to_json.py <input_csv_file> [output_json_file]")
        print("Example: python csv_to_json.py invoice-1.csv invoice-1.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    csv_to_json(input_file, output_file)