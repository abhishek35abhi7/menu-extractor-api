import os
import base64
import tempfile
import re
import json
from flask import Flask, request, jsonify
from google.cloud import vision

app = Flask(__name__)

# For Railway deployment, credentials will be set via environment variable
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    # Create temporary credentials file from environment variable
    import json
    creds_json = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(creds_json, f)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name

client = vision.ImageAnnotatorClient()

def extract_numbers(text):
    """Extract all numbers from text"""
    numbers = []
    
    # Currency symbols with numbers
    currency_matches = re.findall(r'[₹$€£¥](\d+(?:\.\d{2})?)', text)
    numbers.extend([float(m) for m in currency_matches])
    
    # Standalone numbers
    number_matches = re.findall(r'\b(\d{2,4})\b', text)
    numbers.extend([float(n) for n in number_matches])
    
    return numbers

def categorize_item(name):
    """Categorize item as Food or Drink based on name"""
    name_lower = name.lower()
    
    # Drink keywords
    drink_keywords = [
        'beer', 'lager', 'ale', 'stout', 'pilsner', 'hefeweizen', 'wheat', 'ipa',
        'wine', 'sangria', 'champagne', 'prosecco',
        'whiskey', 'whisky', 'bourbon', 'scotch', 'rum', 'vodka', 'gin', 'tequila',
        'cognac', 'brandy', 'liqueur', 'baileys', 'kahlua', 'jagermeister',
        'cocktail', 'martini', 'margarita', 'mojito', 'daiquiri', 'negroni',
        'sour', 'punch', 'sangria', 'bellini', 'cosmopolitan',
        'water', 'soda', 'cola', 'coke', 'pepsi', 'sprite', 'juice', 'coffee', 'tea',
        'milk', 'shake', 'smoothie', 'lemonade', 'iced tea', 'hot chocolate',
        'energy drink', 'sports drink', 'tonic', 'ginger ale', 'club soda',
        'fresh lime', 'cold pressed', 'aerated', 'packaged water', 'perrier',
        'red bull', 'diet coke', 'fresh lime soda', 'fresh lime water'
    ]
    
    for keyword in drink_keywords:
        if keyword in name_lower:
            return "Drink"
    
    return "Food"

def extract_text_from_image(image_data):
    """Extract text from base64 image data"""
    try:
        # Decode base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Create Vision API image object
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f'Vision API error: {response.error.message}')
        
        # Extract text with coordinates
        text_data = []
        for text in response.text_annotations[1:]:
            vertices = text.bounding_poly.vertices
            text_data.append({
                'text': text.description, 
                'x': vertices[0].x, 
                'y': vertices[0].y
            })

        # Sort and group into rows
        text_data.sort(key=lambda x: x['y'])
        rows, current_row = [], []
        row_threshold = 20

        for item in text_data:
            if current_row and abs(item['y'] - current_row[0]['y']) > row_threshold:
                rows.append(current_row)
                current_row = [item]
            else:
                current_row.append(item)
        if current_row:
            rows.append(current_row)

        lines = []
        for row in rows:
            row.sort(key=lambda x: x['x'])
            line_text = ' '.join([i['text'] for i in row])
            lines.append(line_text)

        return lines
        
    except Exception as e:
        raise Exception(f'Error processing image: {str(e)}')

def analyze_menu_structure(lines):
    """Analyze menu structure"""
    analysis = {
        'price_lines': [],
        'name_lines': [],
        'description_lines': [],
        'all_prices': []
    }
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        
        numbers = extract_numbers(line)
        words = line.split()
        
        if numbers and len(words) == 1 and re.match(r'^\d{2,4}$', line):
            analysis['price_lines'].append({
                'index': i,
                'line': line,
                'price': max(numbers)
            })
            analysis['all_prices'].extend(numbers)
            
        elif (line[0].islower() or 
              any(line.lower().startswith(phrase) for phrase in [
                  'a ', 'an ', 'the ', 'with ', 'made ', 'served ', 'crispy ', 
                  'curry ', 'mint ', 'salsa', 'government', 'we levy'
              ]) or
              line.endswith(',') or
              len(words) > 8):
            analysis['description_lines'].append({
                'index': i,
                'line': line
            })
            
        elif (len(line) > 3 and 
              line[0].isupper() and 
              2 <= len(words) <= 8 and
              not re.match(r'^[\d\s₹$€£¥,.-]+$', line)):
            analysis['name_lines'].append({
                'index': i,
                'line': line,
                'numbers': numbers
            })
            analysis['all_prices'].extend(numbers)
    
    return analysis

def clean_name_simple(name):
    """Clean item name"""
    name = re.sub(r'[₹$€£¥]\d+.*$', '', name)
    name = re.sub(r'\s+\d{2,5}\s*$', '', name)
    name = re.sub(r'\s+\d{2,5}\s+\d{2,5}\s*$', '', name)
    name = re.sub(r'\b\d+\s*(ml|cl|oz|inch|cm)\b.*$', '', name, flags=re.I)
    name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
    name = re.sub(r'\s*\[[^]]*\]\s*$', '', name)
    name = ' '.join(name.split()).strip()
    return name

def extract_menu_items(lines):
    """Extract menu items from text lines"""
    analysis = analyze_menu_structure(lines)
    
    if not analysis['all_prices']:
        price_range = [50, 1000]
    else:
        reasonable_prices = [p for p in analysis['all_prices'] if 50 <= p <= 10000]
        if reasonable_prices:
            price_range = [min(reasonable_prices), max(reasonable_prices)]
        else:
            price_range = [50, 1000]
    
    items = []
    
    # Strategy 1: Names with prices on same line
    for name_info in analysis['name_lines']:
        if name_info['numbers']:
            valid_prices = [p for p in name_info['numbers'] if price_range[0] <= p <= price_range[1]]
            if valid_prices:
                clean_name = clean_name_simple(name_info['line'])
                if len(clean_name) > 2:
                    items.append({
                        'name': clean_name,
                        'price': max(valid_prices),
                        'category': categorize_item(clean_name)
                    })
    
    # Strategy 2: Names followed by separate price lines
    used_price_indices = set()
    
    for name_info in analysis['name_lines']:
        if name_info['numbers']:
            continue
            
        name_index = name_info['index']
        clean_name = clean_name_simple(name_info['line'])
        
        if len(clean_name) < 3:
            continue
        
        for price_info in analysis['price_lines']:
            price_index = price_info['index']
            
            if (price_index > name_index and 
                price_index <= name_index + 3 and
                price_index not in used_price_indices):
                
                has_description_between = any(
                    desc['index'] == name_index + 1 
                    for desc in analysis['description_lines']
                )
                
                if (price_index == name_index + 1 or 
                    (has_description_between and price_index == name_index + 2)):
                    
                    items.append({
                        'name': clean_name,
                        'price': price_info['price'],
                        'category': categorize_item(clean_name)
                    })
                    used_price_indices.add(price_index)
                    break
    
    # Remove duplicates
    seen_names = set()
    unique_items = []
    for item in items:
        name_key = item['name'].lower().strip()
        if name_key not in seen_names and len(item['name']) > 2:
            seen_names.add(name_key)
            unique_items.append(item)
    
    return unique_items

@app.route('/extract-menu', methods=['POST'])
def extract_menu():
    try:
        data = request.json
        
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Extract text from image
        lines = extract_text_from_image(data['image_data'])
        
        # Extract menu items
        menu_items = extract_menu_items(lines)
        
        return jsonify({
            'success': True,
            'menu': menu_items,
            'total_items': len(menu_items),
            'food_items': sum(1 for item in menu_items if item['category'] == 'Food'),
            'drink_items': sum(1 for item in menu_items if item['category'] == 'Drink')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Menu Extractor API',
        'endpoints': {
            'health': '/health',
            'extract': '/extract-menu (POST)'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)