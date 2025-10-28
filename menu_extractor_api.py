import os
import base64
import tempfile
import re
import json
from flask import Flask, request, jsonify
from google.cloud import vision

app = Flask(__name__)

# For Render deployment - handle credentials from environment variable
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    try:
        creds_json = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
        # Create temporary credentials file
        creds_path = '/tmp/google-credentials.json'
        with open(creds_path, 'w') as f:
            json.dump(creds_json, f)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
        print("✅ Google credentials configured successfully")
    except Exception as e:
        print(f"❌ Error setting up credentials: {e}")
else:
    print("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")

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
    """Enhanced categorization using comprehensive drink keywords"""
    name_lower = name.lower().strip()
    
    # Remove common prefixes/suffixes that might interfere with matching
    name_clean = re.sub(r'^(fresh|cold|hot|iced|chilled|warm|special|premium|house)\s+', '', name_lower)
    name_clean = re.sub(r'\s+(fresh|cold|hot|iced|chilled|warm|special|premium|house)$', '', name_clean)
    
    # Comprehensive drink keywords list
    drink_keywords = [
        # --- Alcoholic: Beer & Ales ---
        'beer', 'lager', 'ale', 'stout', 'pilsner', 'porter', 'hefeweizen', 'wheat',
        'ipa', 'double ipa', 'session ipa', 'craft beer', 'draught beer', 'draft beer',
        'amber ale', 'brown ale', 'pale ale', 'blonde ale', 'saison', 'kolsch',
        # --- Alcoholic: Wines & Sparkling ---
        'wine', 'red wine', 'white wine', 'rose', 'dessert wine', 'fortified wine',
        'sangria', 'champagne', 'prosecco', 'cava', 'sparkling wine',
        # --- Alcoholic: Spirits ---
        'whiskey', 'whisky', 'bourbon', 'scotch', 'rye whiskey', 'irish whiskey',
        'rum', 'vodka', 'gin', 'tequila', 'mezcal', 'cognac', 'brandy', 'armagnac',
        'absinthe', 'soju', 'sake', 'grappa', 'baijiu',
        # --- Alcoholic: Liqueurs & Creams ---
        'liqueur', 'amaretto', 'baileys', 'kahlua', 'jagermeister', 'cointreau',
        'grand marnier', 'triple sec', 'midori', 'chambord', 'drambuie', 'sambuca',
        'limoncello', 'frangelico', 'curacao', 'campari', 'aperol', 'vermouth',
        # --- Alcoholic: Cocktails ---
        'cocktail', 'martini', 'margarita', 'mojito', 'daiquiri', 'negroni', 'cosmopolitan',
        'old fashioned', 'manhattan', 'bloody mary', 'whiskey sour', 'mai tai', 'pina colada',
        'tequila sunrise', 'gin tonic', 'vodka tonic', 'rum punch', 'bellini', 'spritz',
        'mint julep', 'paloma', 'espresso martini', 'long island iced tea', 'blue lagoon',
        # --- Non-Alcoholic: Water & Sodas ---
        'water', 'sparkling water', 'mineral water', 'packaged water', 'perrier', 'aerated',
        'soda', 'club soda', 'tonic', 'ginger ale', 'cola', 'coke', 'pepsi', 'sprite',
        'mountain dew', 'fanta', '7up', 'diet coke', 'fresh lime', 'fresh lime soda',
        'fresh lime water', 'cold pressed', 'carbonated drink',
        # --- Non-Alcoholic: Juices & Smoothies ---
        'juice', 'orange juice', 'apple juice', 'mango juice', 'pineapple juice',
        'pomegranate juice', 'grape juice', 'watermelon juice', 'cranberry juice',
        'lime juice', 'lemonade', 'fresh juice', 'mocktail', 'fruit punch',
        'smoothie', 'milkshake', 'protein shake', 'cold pressed juice', 'lassi',
        'buttermilk', 'aam panna', 'jaljeera', 'kokum sherbet', 'rose milk', 'falooda',
        # --- Café: Coffee & Espresso Variants ---
        'coffee', 'black coffee', 'americano', 'espresso', 'double espresso',
        'ristretto', 'lungo', 'cappuccino', 'latte', 'flat white', 'macchiato',
        'café mocha', 'iced coffee', 'cold brew', 'nitro cold brew', 'affogato',
        'frappuccino', 'turkish coffee', 'filter coffee', 'south indian filter coffee',
        'madras coffee', 'instant coffee', 'decaf', 'espresso shot',
        # --- Café: Hot Chocolate & Specialty Drinks ---
        'hot chocolate', 'chocolate milk', 'white chocolate mocha', 'caramel macchiato',
        'hazelnut latte', 'vanilla latte', 'irish coffee', 'spiced coffee', 'chai latte',
        # --- Tea: Hot, Iced & Infused ---
        'tea', 'black tea', 'green tea', 'white tea', 'oolong tea', 'herbal tea',
        'masala chai', 'ginger tea', 'lemon tea', 'peppermint tea', 'hibiscus tea',
        'earl grey', 'english breakfast', 'chamomile tea', 'jasmine tea',
        'iced tea', 'lemon iced tea', 'peach iced tea', 'mint tea', 'matcha', 'matcha latte',
        'bubble tea', 'boba', 'thai tea', 'milk tea', 'taro milk tea',
        # --- Energy & Sports ---
        'energy drink', 'red bull', 'monster', 'gatorade', 'powerade', 'sports drink',
        'isotonic drink', 'electrolyte water', 'sting',
        # --- Indian Traditional Beverages ---
        'sweet lassi', 'salted lassi', 'mango lassi', 'chaas', 'thandai', 'badam milk',
        'kesar milk', 'haldi doodh', 'turmeric milk', 'rabri', 'milk sharbat',
        'nannari sherbet', 'malai shake', 'dry fruit shake', 'sattu drink',
        'bel sherbet', 'solkadhi', 'neera', 'toddy', 'tender coconut water',
        'nimbu pani', 'piyush', 'chaach', 'kanji', 'ragi malt', 'ragi ambli',
        'sattu sharbat', 'amla juice',
        # --- Indian Tea & Infusions ---
        'chai', 'adrak chai', 'elaichi chai', 'cardamom tea', 'tulsi tea',
        'green chai', 'cutting chai', 'kulhad chai', 'irani chai', 'sulaimani chai',
        'kahwa', 'nilgiri tea', 'assam tea', 'darjeeling tea', 'kesar chai',
        'rose chai', 'kashmiri chai',
        # --- Indian Regional Drinks ---
        'kulukki sarbath', 'nannari sarbath', 'paneer soda', 'rose soda', 'jeera soda',
        'masala soda', 'banta', 'goli soda', 'lemon soda', 'imli soda', 'falsa sherbet',
        'chandan sharbat', 'jamun juice', 'mosambi juice', 'kala khatta',
        'tamarind drink', 'panakam', 'bel juice', 'barley water',
        # --- Indian Fusion & Café Style ---
        'kulfi shake', 'mango shake', 'strawberry shake', 'oreo shake', 'kitkat shake',
        'banana shake', 'coffee shake', 'butterscotch shake', 'rose frappe',
        'thick shake', 'detox juice', 'matcha chai', 'turmeric latte',
        # --- Packaged & Modern Beverages ---
        'bisleri', 'kinley', 'aquafina', 'himalayan water', 'appfizz', 'bovonto',
        'thumbs up', 'coca cola', 'maaza', 'frooti', 'slice', 'real juice',
        'paper boat', 'tropicana', 'mirinda', 'paperboat drink',
        # --- Festival & Ritual ---
        'bhang thandai', 'panchamrit', 'charanamrit', 'sandalwood milk',
        'holi thandai', 'navratri drink', 'diwali sharbat'
    ]
    
    # Create a set for faster lookups
    drink_keywords_set = set(drink_keywords)
    
    # Food keywords that should never be drinks (to prevent false positives)
    food_keywords = [
        'soup', 'broth', 'curry', 'gravy', 'sauce', 'dal', 'rasam', 'sambar',
        'bread', 'roti', 'naan', 'chapati', 'paratha', 'rice', 'biryani', 'pulao',
        'salad', 'starter', 'appetizer', 'main course', 'dessert', 'sweet',
        'ice cream', 'kulfi', 'gulab jamun', 'rasgulla', 'kheer', 'halwa',
        'pizza', 'burger', 'sandwich', 'wrap', 'roll', 'pasta', 'noodles',
        'chicken', 'mutton', 'fish', 'paneer', 'egg', 'vegetable', 'sabzi'
    ]
    
    # Check for explicit food keywords first (prevents false drink classification)
    for keyword in food_keywords:
        if keyword in name_clean:
            return "Food"
    
    # Check for drink keywords using exact and partial matches
    # 1. Exact match check
    if name_clean in drink_keywords_set:
        return "Drink"
    
    # 2. Partial match check (keyword appears in the name)
    for keyword in drink_keywords:
        if keyword in name_clean:
            return "Drink"
    
    # 3. Pattern-based detection for drinks
    drink_patterns = [
        r'\b\d+\s*ml\b',           # Volume indicators: 250ml, 500ml
        r'\b\d+\s*cl\b',           # Centiliters: 33cl
        r'\b\d+\s*oz\b',           # Ounces: 12oz
        r'\bbottle\b',             # Bottle indicators
        r'\bcan\b',                # Can indicators  
        r'\bglass\b(?!\s+noodles)', # Glass (but not glass noodles)
        r'\bpeg\b',                # Alcohol peg
        r'\bshot\b(?!\s+glass)',   # Shot (but not shot glass as food item)
        r'\bmug\b',                # Mug (coffee/tea context)
        r'\bcup\b(?!\s+noodles)',  # Cup (but not cup noodles)
    ]
    
    for pattern in drink_patterns:
        if re.search(pattern, name_clean, re.IGNORECASE):
            return "Drink"
    
    # 4. Check for common drink suffixes/prefixes that weren't caught
    drink_indicators = [
        'drink', 'beverage', 'liquid', 'refresher', 'cooler', 'slush',
        'shake', 'smoothie', 'juice', 'water', 'soda', 'tea', 'coffee',
        'milk', 'lassi', 'sherbet', 'sharbat', 'punch'
    ]
    
    for indicator in drink_indicators:
        if indicator in name_clean:
            return "Drink"
    
    # Default to Food for ambiguous cases
    return "Food"

def extract_text_from_image(image_data):
    """Extract text from base64 image data"""
    try:
        print(f"📥 Received image data length: {len(image_data)}")
        
        # Remove data URI prefix if present
        if ',' in image_data:
            parts = image_data.split(',', 1)
            print(f"🔍 Data URI prefix: {parts[0][:50]}...")
            image_data = parts[1]
        
        # Clean the base64 string - remove whitespace and newlines
        original_length = len(image_data)
        image_data = image_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        print(f"🧹 Cleaned base64 string: {original_length} -> {len(image_data)} chars")
        
        # Add padding if necessary
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += '=' * (4 - missing_padding)
            print(f"➕ Added {4 - missing_padding} padding characters")
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
            print(f"✅ Successfully decoded base64 to {len(image_bytes)} bytes")
        except Exception as decode_error:
            print(f"❌ Base64 decode failed: {str(decode_error)}")
            raise Exception(f'Base64 decode error: {str(decode_error)}')
        
        # Validate image size
        if len(image_bytes) < 100:
            raise Exception(f'Image data too small ({len(image_bytes)} bytes) - likely corrupted')
        
        # Check if it looks like a valid image (basic magic number check)
        magic_numbers = {
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x89\x50\x4E\x47': 'PNG',
            b'\x47\x49\x46': 'GIF',
            b'\x42\x4D': 'BMP'
        }
        
        detected_format = None
        for magic, format_name in magic_numbers.items():
            if image_bytes.startswith(magic):
                detected_format = format_name
                break
        
        if detected_format:
            print(f"🖼️ Detected image format: {detected_format}")
        else:
            print(f"⚠️ Unknown image format. First bytes: {image_bytes[:10].hex()}")
        
        # Create Vision API image object directly
        print("🔄 Sending to Google Vision API...")
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        
        if response.error.message:
            print(f"❌ Vision API error: {response.error.message}")
            raise Exception(f'Vision API error: {response.error.message}')
        
        print(f"✅ Vision API returned {len(response.text_annotations)} text annotations")
        
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

        print(f"📝 Extracted {len(lines)} text lines")
        return lines
        
    except Exception as e:
        print(f"❌ Error in extract_text_from_image: {str(e)}")
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
        print(f"❌ Error in /extract-menu: {str(e)}")
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
