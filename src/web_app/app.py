"""
CaloScan - Food Nutrition Analysis Web App
Supports multiple models:
- GPT-2 based nutrition model (CaloScan v1)
- EfficientNet-B3 food classifier with USDA nutrition lookup
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import os
import json
import re
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from effnet_classifier import EfficientNetClassifier, get_nutrition_from_usda

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['USDA_API_KEY'] = os.environ.get('USDA_API_KEY', '')  # Set your USDA API key

# Global variables for models
gpt2_model = None
gpt2_tokenizer = None
effnet_model = None
device = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load all available models"""
    global gpt2_model, gpt2_tokenizer, effnet_model, device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models_loaded = {}

    # Load GPT-2 model (CaloScan v1)
    try:
        print("Loading GPT-2 nutrition model...")
        model_path = r"E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125"

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_path)

        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
            gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id

        gpt2_model.to(device)
        gpt2_model.eval()

        print(f"GPT-2 model loaded successfully! ({gpt2_model.num_parameters():,} parameters)")
        models_loaded['caloscan-v1'] = True
    except Exception as e:
        print(f"Error loading GPT-2 model: {e}")
        models_loaded['caloscan-v1'] = False

    # Load EfficientNet-B3 classifier
    try:
        print("Loading EfficientNet-B3 classifier...")
        effnet_path = r"E:\Startup\CaloScan\src\finetuned-food\food_classifier_effb3\food_classifier_effb3_scripted.pt"

        effnet_model = EfficientNetClassifier(effnet_path, device)
        if effnet_model.load_model():
            print("EfficientNet-B3 model loaded successfully!")
            models_loaded['effnet-b3'] = True
        else:
            models_loaded['effnet-b3'] = False
    except Exception as e:
        print(f"Error loading EfficientNet-B3 model: {e}")
        models_loaded['effnet-b3'] = False

    return any(models_loaded.values()), models_loaded

def analyze_image_gpt2(image_path, dish_name="", ingredients="", cooking_method="", portion_size=""):
    """
    Analyze food image using GPT-2 model and predict nutritional values
    """
    global gpt2_model, gpt2_tokenizer, device

    # If fields are empty, use defaults
    if not dish_name:
        dish_name = "Unknown Dish"
    if not ingredients:
        ingredients = "[]"
    if not cooking_method:
        cooking_method = "Unknown"
    if not portion_size:
        portion_size = "Standard serving"

    # Create instruction prompt
    instruction = "Analyze the given food image and return the nutritional profile in JSON (Calories, Protein, Fat, Carbs, Fiber) and short human summary."

    # Create input text
    input_text = f"""Image URL: {image_path}
Dish: {dish_name}
Ingredients: {ingredients}
Cooking method: {cooking_method}
Portion size: {portion_size}"""

    # Generate prediction
    prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"

    inputs = gpt2_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs["input_ids"],
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.pad_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id
        )

    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract output
    if "Output:" in generated_text:
        prediction = generated_text.split("Output:")[-1].strip()
    else:
        prediction = generated_text

    # Parse JSON from prediction
    nutrition_data = parse_nutrition_json(prediction)

    return {
        'prediction_text': prediction,
        'nutrition': nutrition_data,
        'input_data': {
            'dish_name': dish_name,
            'ingredients': ingredients,
            'cooking_method': cooking_method,
            'portion_size': portion_size
        }
    }


def analyze_image_effnet(image_path, portion_g=200):
    """
    Analyze food image using EfficientNet-B3 classifier and USDA nutrition data
    """
    global effnet_model

    try:
        # Get food classification predictions
        predictions = effnet_model.predict(image_path, top_k=5)

        if not predictions:
            raise Exception("No predictions returned from model")

        # Get the top prediction
        top_prediction = predictions[0]
        food_name = top_prediction['class']

        # Get nutrition data from USDA
        nutrition_data = get_nutrition_from_usda(
            food_name,
            app.config['USDA_API_KEY'],
            portion_g=portion_g
        )

        # Create prediction text
        prediction_lines = []
        prediction_lines.append(f"Detected Food: {top_prediction['class_display']}")
        prediction_lines.append(f"Confidence: {top_prediction['probability']*100:.1f}%")
        prediction_lines.append(f"\nTop 5 Predictions:")
        for i, pred in enumerate(predictions, 1):
            prediction_lines.append(f"  {i}. {pred['class_display']} ({pred['probability']*100:.1f}%)")

        prediction_lines.append(f"\nNutritional Information (per {portion_g}g):")
        prediction_lines.append(f"  Calories: {nutrition_data['calories']:.1f} kcal")
        prediction_lines.append(f"  Protein: {nutrition_data['protein']:.1f} g")
        prediction_lines.append(f"  Fat: {nutrition_data['fat']:.1f} g")
        prediction_lines.append(f"  Carbohydrates: {nutrition_data['carbohydrates']:.1f} g")
        prediction_lines.append(f"  Fiber: {nutrition_data['fiber']:.1f} g")
        prediction_lines.append(f"\nSource: {nutrition_data['source']}")

        prediction_text = "\n".join(prediction_lines)

        return {
            'prediction_text': prediction_text,
            'nutrition': nutrition_data,
            'predictions': predictions,
            'top_prediction': top_prediction,
            'portion_size': f"{portion_g}g",
            'input_data': {
                'dish_name': top_prediction['class_display'],
                'portion_size': f"{portion_g}g"
            }
        }

    except Exception as e:
        raise Exception(f"Error analyzing image with EfficientNet: {str(e)}")

def parse_nutrition_json(text):
    """
    Extract nutrition values from JSON text
    """
    try:
        # Try to find the first JSON object in the text
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            json_str = json_match.group(0)
            nutrition = json.loads(json_str)

            # Standardize the keys
            result = {
                'calories': nutrition.get('calories_kcal', nutrition.get('calories', 0)),
                'protein': nutrition.get('protein_g', nutrition.get('protein', 0)),
                'fat': nutrition.get('fat_g', nutrition.get('fat', 0)),
                'carbohydrates': nutrition.get('carbohydrate_g', nutrition.get('carbohydrates', 0)),
                'fiber': nutrition.get('fiber_g', nutrition.get('fiber', 0))
            }
            return result
    except Exception as e:
        print(f"Error parsing JSON: {e}")

    # Return default values if parsing fails
    return {
        'calories': 0,
        'protein': 0,
        'fat': 0,
        'carbohydrates': 0,
        'fiber': 0
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image upload and analysis"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get form data
        selected_model = request.form.get('model', 'caloscan-v1')
        dish_name = request.form.get('dish_name', '')
        ingredients = request.form.get('ingredients', '')
        cooking_method = request.form.get('cooking_method', '')
        portion_size = request.form.get('portion_size', '')

        # Route to appropriate model
        if selected_model == 'effnet-b3':
            # Parse portion size (default 200g if not specified)
            portion_g = 200
            if portion_size:
                # Try to extract grams from portion_size string
                import re
                match = re.search(r'(\d+)\s*g', portion_size)
                if match:
                    portion_g = int(match.group(1))

            result = analyze_image_effnet(filepath, portion_g=portion_g)
        else:
            # Use GPT-2 model (default)
            result = analyze_image_gpt2(
                filepath,
                dish_name=dish_name,
                ingredients=ingredients,
                cooking_method=cooking_method,
                portion_size=portion_size
            )

        # Add metadata to result
        result['image_path'] = filepath.replace('\\', '/')
        result['model_used'] = selected_model

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models': {
            'gpt2': gpt2_model is not None,
            'effnet-b3': effnet_model is not None
        },
        'device': str(device)
    })

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Load the models
    success, models_status = load_models()

    if success:
        print("\n" + "="*80)
        print("CaloScan Web App Starting...")
        print("="*80)
        print("\nModels loaded:")
        print(f"  - GPT-2 (CaloScan v1): {'OK' if models_status.get('caloscan-v1') else 'FAILED'}")
        print(f"  - EfficientNet-B3: {'OK' if models_status.get('effnet-b3') else 'FAILED'}")
        print("\nOpen your browser and navigate to:")
        print("  http://localhost:5000")
        print("\nPress CTRL+C to stop the server")
        print("="*80 + "\n")

        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load any models. Please check the model paths.")
